import blenderproc as bproc  # must be line 1 under `blenderproc run`
# Render one scene spec with BlenderProc: physics-drop the food, then RGB, z-depth and instance ids.
#
#     blenderproc run [--custom-blender-path ~/blender/blender-4.2.1-linux-x64] \
#         synth/render.py SCENE_DIR/spec.json RAW_OUT_DIR [--spp 64]
#
# Runs inside Blender's Python (numpy, bpy, blenderproc only). One scene per process,
# because BlenderProc output passes can't be re-registered and Blender leaks memory
# over long runs. Writes RAW_OUT_DIR/frame_NNNNNN.npz (rgb uint8, depth float32 z-depth
# in metres, seg uint8: 0 table/background, 1..K items, 254 container) and
# render_meta.json (final item poses and the volume of each mesh as placed in the
# scene, a check on the spec's ground truth). synth/finalize.py does the rest.


import argparse
import json
import sys
import time
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

CONTAINER_ID = 254
DISTRACTOR_ID = 253


def mesh_from_npz(path: Path, name: str):
    z = np.load(path)
    obj = bproc.object.create_with_empty_mesh(name)
    me = obj.get_mesh()
    me.from_pydata(z["V"].astype(float).tolist(), [], z["F"].tolist())
    me.validate()
    me.update()
    return obj


def load_gltf(p: Path):
    """Poly Haven glTF -> one mesh object in Blender's Z-up frame with its node transforms
    baked in (the importer converts Y up; empties / parents are dropped)."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(p))
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    for o in meshes:
        M = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = M
    for o in new:
        if o.type != "MESH":
            bpy.data.objects.remove(o, do_unlink=True)
    objs = [bproc.types.MeshObject(o) for o in meshes]
    obj = objs[0]
    if len(objs) > 1:
        obj.join_with_other_objects(objs[1:])
    obj.persist_transformation_into_mesh()
    return obj


def drop_stray_leaves(obj, per_leaf: int, keep_r: float) -> int:
    """Delete leaves (consecutive blocks of per_leaf vertices) that slid off the dish: centre
    beyond keep_r of the dish axis or resting on the table. A real served salad stays on the
    plate; strays were the most common QC failure. Keeps at least the most central leaf."""
    me = obj.blender_obj.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    M = np.array(obj.blender_obj.matrix_world)
    V = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    n = len(V) // per_leaf
    ctr = V[: n * per_leaf].reshape(n, per_leaf, 3)
    r = np.hypot(ctr[:, :, 0].mean(1), ctr[:, :, 1].mean(1))
    zmin = ctr[:, :, 2].min(1)
    bad = (r > keep_r) | (zmin < 0.002)
    if bad.all():
        bad[np.argmin(r)] = False
    if not bad.any():
        return 0
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    gone = [bm.verts[i] for k in np.flatnonzero(bad) for i in range(k * per_leaf, (k + 1) * per_leaf)]
    bmesh.ops.delete(bm, geom=gone, context="VERTS")
    bm.to_mesh(me)
    bm.free()
    return int(bad.sum())


def simulate_cloth(cloth: list, obstacles: list, keep_r: float, frames: int = 150) -> None:
    """Drop thin sheets (leaves) as cloth onto the settled scene, freeze the last frame and give
    them thickness: they drape over food and the dish and pile on each other (self collision).
    Leaves that slid off the dish (beyond keep_r of its axis) are removed."""
    scene = bpy.context.scene
    for o in obstacles:
        m = o.blender_obj.modifiers.new("synth_collision", "COLLISION")
        o.blender_obj.collision.thickness_outer = 0.001
        o.blender_obj.collision.cloth_friction = 20.0   # damp leaves grip food; at 8 they slid off domes
    for it, o in cloth:
        m = o.blender_obj.modifiers.new("synth_cloth", "CLOTH")
        s = m.settings
        # settings from a stability sweep (cm-scale cloth): lighter vertices barely fall against
        # air drag, larger contact distances / pointed leaf tips made self-collision explode
        s.quality = 15
        s.mass = 0.003                                  # kg per vertex
        s.tension_stiffness = s.compression_stiffness = s.shear_stiffness = 15.0
        s.bending_stiffness = it["cloth"]["bending"]   # crisp lettuce vs limp spinach
        s.air_damping = 0.5                             # 2.0 left leaves hanging mid-air after 120 frames
        cs = m.collision_settings
        cs.use_self_collision = True
        cs.distance_min = 0.0006
        cs.self_distance_min = 0.0006
        cs.collision_quality = 4
        cs.friction = 20.0
        m.point_cache.frame_start, m.point_cache.frame_end = 1, frames
    scene.frame_start, scene.frame_end = 1, frames
    for f in range(1, frames + 1):
        scene.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    for it, o in cloth:
        me = bpy.data.meshes.new_from_object(o.blender_obj.evaluated_get(dg))
        o.blender_obj.modifiers.clear()
        o.blender_obj.data = me
        grid = it["cloth"].get("grid", (12, 5))
        n_gone = drop_stray_leaves(o, grid[0] * grid[1], keep_r)
        if n_gone:
            print(f"[render] {it['name']}: removed {n_gone} leaves that slid off the dish", flush=True)
        # the 12 x 5 cloth grid is coarse: subdivide so leaves curve instead of showing facets
        # and sawtooth margins, then shade smooth
        sub = o.blender_obj.modifiers.new("synth_subsurf", "SUBSURF")
        sub.levels = sub.render_levels = 2
        for poly in o.blender_obj.data.polygons:
            poly.use_smooth = True
        sol = o.blender_obj.modifiers.new("synth_solid", "SOLIDIFY")
        sol.thickness, sol.offset, sol.use_rim = it["cloth"]["thickness_m"], 0.0, True
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(o.blender_obj.evaluated_get(dg))
        o.blender_obj.modifiers.clear()
        o.blender_obj.data = me
    for o in obstacles:
        mod = o.blender_obj.modifiers.get("synth_collision")
        if mod:
            o.blender_obj.modifiers.remove(mod)
    scene.frame_set(0)
    scene.frame_start, scene.frame_end = 0, 0        # BlenderProc appends camera poses from frame_end


def collision_parts(p: str) -> list:
    """synth.collision cache: V (all part vertices), n (vertices per part)."""
    z = np.load(p)
    return np.split(z["V"].astype(float), np.cumsum(z["n"])[:-1])


def compound_collider(obj, parts: list, c_local: np.ndarray, mass: float | None, damping: float = 0.0) -> list:
    """Compound rigid body: hidden convex-hull children (one per CoACD part, in the object's
    mesh-local frame shifted by its origin move c_local); the visible mesh itself is not
    simulated, so its resolution doesn't matter. mass None: the children only (cloth obstacles)."""
    kids = []
    for k, P in enumerate(parts):
        bm = bmesh.new()
        for p in np.asarray(P, float) - c_local:
            bm.verts.new(p)
        bmesh.ops.convex_hull(bm, input=bm.verts)
        kid = bproc.object.create_with_empty_mesh(f"{obj.get_name()}_part{k}")
        bm.to_mesh(kid.get_mesh())
        bm.free()
        kid.blender_obj.parent = obj.blender_obj
        kid.blender_obj.matrix_parent_inverse.identity()
        if mass is not None:
            kid.enable_rigidbody(True, collision_shape="CONVEX_HULL", collision_margin=0.0005)
        kid.hide()
        kids.append(kid)
    if mass is not None:
        obj.enable_rigidbody(True, collision_shape="COMPOUND", mass=mass, friction=0.8, linear_damping=damping,
                         angular_damping=damping, collision_margin=0.0005)
    return kids


def place_on_table(obj, location, euler) -> None:
    """Static prop: pose it, then lift it so its lowest point touches the table (z = 0)."""
    obj.set_location(location)
    obj.set_rotation_euler(euler)
    bpy.context.view_layer.update()
    zmin = float(np.array(obj.get_bound_box())[:, 2].min())
    loc = np.array(obj.get_location())
    obj.set_location([loc[0], loc[1], loc[2] - zmin + 0.0003])


def load_mesh(item: dict, scene_dir: Path):
    p = Path(item["mesh"])
    if not p.is_absolute():
        p = scene_dir / p
    if item["fmt"] == "npz":
        return mesh_from_npz(p, item["name"])
    if item["fmt"] in ("gltf", "glb"):
        return load_gltf(p)
    up = item.get("up_axis", "Y")      # OBJ convention is Y up; MetaFood3D scans are Z up
    fwd = "Y" if up == "Z" else "NEGATIVE_Z"
    objs = bproc.loader.load_obj(str(p), forward_axis=fwd, up_axis=up)
    obj = objs[0]
    if len(objs) > 1:
        obj.join_with_other_objects(objs[1:])
    return obj


def flat_material(obj, name: str, color, roughness: float):
    mat = bproc.material.create(name)
    mat.set_principled_shader_value("Base Color", [*color, 1.0])
    mat.set_principled_shader_value("Roughness", float(roughness))
    obj.replace_materials(mat)


def textured_material(obj, name: str, color, roughness: float, scale: float, metallic: float = 0.0):
    """Principled BSDF whose colour and bump follow 3D noise in object space (metres),
    so procedural food isn't a flat colour: scale ~300 gives mm-sized grain, ~60 a sauce."""
    mat = bproc.material.create(name)
    bsdf = mat.get_the_one_node_with_type("BsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = float(roughness)
    bsdf.inputs["Metallic"].default_value = float(metallic)
    coord = mat.nodes.new("ShaderNodeTexCoord")
    noise = mat.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = float(scale)
    noise.inputs["Detail"].default_value = 8.0
    ramp = mat.nodes.new("ShaderNodeValToRGB")
    c = np.asarray(color, float)
    ramp.color_ramp.elements[0].color = [*np.clip(c * 0.7, 0, 1), 1.0]
    ramp.color_ramp.elements[1].color = [*np.clip(c * 1.2, 0, 1), 1.0]
    bump = mat.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.002
    mat.links.new(coord.outputs["Object"], noise.inputs["Vector"])
    mat.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    mat.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    mat.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    mat.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    obj.replace_materials(mat)


def food_material(obj, name: str, color, roughness: float, L: dict) -> None:
    """Procedural food by consistency (synth.looks). Voronoi cells in object space (metres) are
    grains (rounded, dark crevices), oat flakes (pale plateaus in paste) or chunks in a liquid;
    low-frequency noise and distorted bands are lumps and spoon swirls. Wet foods get a clear
    coat and subsurface scattering, so oatmeal and stews read as glossy paste, not sand."""
    mat = bproc.material.create(name)
    nt, links = mat.nodes, mat.links
    bsdf = mat.get_the_one_node_with_type("BsdfPrincipled")
    c = np.asarray(color, float)

    def node(kind, **inp):
        n = nt.new(kind)
        for k, v in inp.items():
            n.inputs[k].default_value = v
        return n

    def ramp(src, a, b, lo=0.0, hi=1.0):
        """smoothstep from (a -> lo) to (b -> hi), a < b"""
        n = node("ShaderNodeMapRange", **{"From Min": a, "From Max": b, "To Min": lo, "To Max": hi})
        n.interpolation_type = "SMOOTHSTEP"
        links.new(src, n.inputs["Value"])
        return n.outputs["Result"]

    def math(op, a, b):
        n = nt.new("ShaderNodeMath")
        n.operation = op
        for i, v in enumerate((a, b)):
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = float(v)
            else:
                links.new(v, n.inputs[i])
        return n.outputs[0]

    def mix(fac, a, b):
        n = nt.new("ShaderNodeMixRGB")
        for sock, v in ((n.inputs["Fac"], fac), (n.inputs["Color1"], a), (n.inputs["Color2"], b)):
            if isinstance(v, (int, float)):
                sock.default_value = float(v)
            elif isinstance(v, (list, tuple, np.ndarray)):
                sock.default_value = [*np.clip(v, 0, 1), 1.0]
            else:
                links.new(v, sock)
        return n.outputs["Color"]

    def vec(op, a, b):
        n = nt.new("ShaderNodeVectorMath")
        n.operation = op
        links.new(a, n.inputs[0])
        if isinstance(b, (int, float)):
            n.inputs["Scale"].default_value = float(b)
        elif isinstance(b, tuple):
            n.inputs[1].default_value = b
        else:
            links.new(b, n.inputs[1])
        return n.outputs["Vector"]

    coord = nt.new("ShaderNodeTexCoord").outputs["Object"]
    tv = L.get("tint_var", 0.06)
    patch = node("ShaderNodeTexNoise", Scale=25.0, Detail=3.0)      # darker / lighter patches
    links.new(coord, patch.inputs["Vector"])
    colour = mix(ramp(patch.outputs["Fac"], 0.3, 0.7), (c * (1 - tv)).tolist(), (c * (1 + tv)).tolist())
    height = None

    cells = L.get("cells")
    if cells:
        # domain warp so cells aren't regular polygons
        warp = node("ShaderNodeTexNoise", Scale=float(cells["scale"]) / 3, Detail=2.0)
        links.new(coord, warp.inputs["Vector"])
        off = vec("SCALE", vec("SUBTRACT", warp.outputs["Color"], (0.5, 0.5, 0.5)), cells["distort"] / cells["scale"])
        p = vec("ADD", coord, off)
        vor = nt.new("ShaderNodeTexVoronoi")
        vor.feature = "F1"
        vor.inputs["Scale"].default_value = float(cells["scale"])
        links.new(p, vor.inputs["Vector"])
        vid = nt.new("ShaderNodeTexVoronoi")                 # same cells: their random colour
        vid.inputs["Scale"].default_value = float(cells["scale"])
        links.new(p, vid.inputs["Vector"])
        rnd = nt.new("ShaderNodeSeparateColor")
        links.new(vid.outputs["Color"], rnd.inputs["Color"])
        d = vor.outputs["Distance"]
        if cells["mode"] == "dome":                          # rounded grain / curd, crevice at the edge
            height = ramp(d, 0.05, 0.55, 1.0, 0.0)
        elif cells["mode"] == "flake":                       # some cells are flakes, raised soft plateaus in paste
            is_flake = math("LESS_THAN", rnd.outputs["Blue"], cells.get("cover", 0.55))
            height = math("MULTIPLY", ramp(d, 0.3, 0.5, 1.0, 0.0), is_flake)
        else:                                                # chunks where F1 is small, liquid between
            r = 0.2 + 0.35 * cells.get("cover", 0.4)
            height = ramp(d, r - 0.12, r, 1.0, 0.0)
        j = cells["jitter"]
        colour = mix(0.5, colour, mix(rnd.outputs["Red"], (c * (1 - j)).tolist(), (c * (1 + j)).tolist()))
        if cells["mode"] == "chunk" and L.get("chunk_color") is not None:
            cc = np.asarray(L["chunk_color"])
            colour = mix(height, colour, mix(rnd.outputs["Red"], (cc * 0.8).tolist(), (cc * 1.2).tolist()))
        if cells["mode"] == "flake":                         # oat flakes paler and drier than the paste
            colour = mix(height, colour, mix(rnd.outputs["Red"], (c * 1.05).tolist(), (c * 1.3).tolist()))
        lo = 0.0
        for sc, frac in L.get("speckles", []):               # peas / carrot / egg in fried rice, red quinoa
            m = math("MULTIPLY", math("GREATER_THAN", rnd.outputs["Green"], lo),
                     math("LESS_THAN", rnd.outputs["Green"], lo + frac))
            colour = mix(m, colour, list(sc))
            lo += frac
        if cells.get("crevice"):
            colour = mix(math("MULTIPLY", math("SUBTRACT", 1.0, height), cells["crevice"]), colour,
                         (c * 0.45).tolist())

    # solids in a wet food (beans, oat flakes) are matte; only the liquid between them shines
    solid = height if cells and cells["mode"] in ("chunk", "flake") else None
    lump_scale, lump_w = L.get("lumps", (50.0, 0.15))
    lump = node("ShaderNodeTexNoise", Scale=float(lump_scale), Detail=4.0)
    links.new(coord, lump.inputs["Vector"])
    h = math("MULTIPLY", lump.outputs["Fac"], lump_w)
    height = h if height is None else math("ADD", height, h)
    if L.get("swirl"):                                       # spoon swirls / ripples
        sw_scale, sw_w = L["swirl"]
        wave = node("ShaderNodeTexWave", Scale=float(sw_scale), Distortion=8.0, Detail=2.0)
        wave.wave_profile = "SIN"
        links.new(coord, wave.inputs["Vector"])
        height = math("ADD", height, math("MULTIPLY", wave.outputs["Fac"], sw_w))

    bump = node("ShaderNodeBump", Strength=float(L.get("bump", 0.4)), Distance=0.0015)
    links.new(height, bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    if L.get("ao"):                    # paste between scattered grains sits in their shadow
        ao = nt.new("ShaderNodeAmbientOcclusion")
        ao.inputs["Distance"].default_value = 0.004
        ao.samples = 8
        shade = nt.new("ShaderNodeMapRange")
        shade.inputs["To Min"].default_value = 0.35
        links.new(ao.outputs["AO"], shade.inputs["Value"])
        dark = nt.new("ShaderNodeMixRGB")
        dark.blend_type, dark.inputs["Fac"].default_value = "MULTIPLY", 1.0
        links.new(colour, dark.inputs["Color1"])
        links.new(shade.outputs["Result"], dark.inputs["Color2"])
        colour = dark.outputs["Color"]
    links.new(colour, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = float(roughness)
    bsdf.inputs["Subsurface Weight"].default_value = float(L.get("sss", 0.2))
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.6, 0.35)
    bsdf.inputs["Subsurface Scale"].default_value = 0.002          # metres: food scatters over ~mm
    bsdf.inputs["Coat Weight"].default_value = float(L.get("coat", 0.0))
    bsdf.inputs["Coat Roughness"].default_value = 0.08
    if solid is not None:
        links.new(math("ADD", float(roughness), math("MULTIPLY", solid, 0.35)), bsdf.inputs["Roughness"])
        links.new(math("MULTIPLY", float(L.get("coat", 0.0)), math("SUBTRACT", 1.0, solid)), bsdf.inputs["Coat Weight"])
    obj.replace_materials(mat)


def pbr_table(obj, tab: dict) -> None:
    """Photographed tabletop (Poly Haven colour / roughness / OpenGL normal maps) in object
    space at its real-world tile size, rotated, offset and slightly tinted per scene."""
    P = tab["pbr"]
    mat = bproc.material.create("table")
    nt, links = mat.nodes, mat.links
    bsdf = mat.get_the_one_node_with_type("BsdfPrincipled")
    coord = nt.new("ShaderNodeTexCoord")
    mapping = nt.new("ShaderNodeMapping")
    mapping.inputs["Location"].default_value = (P["offset"][0], P["offset"][1], 0.0)
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, P["rotation"])
    s = 1.0 / P["tile_m"]
    mapping.inputs["Scale"].default_value = (s, s, s)
    links.new(coord.outputs["Object"], mapping.inputs["Vector"])

    def tex(name: str, colour: bool):
        t = nt.new("ShaderNodeTexImage")
        t.image = bpy.data.images.load(str(Path(P["dir"]) / name), check_existing=True)
        t.image.colorspace_settings.name = "sRGB" if colour else "Non-Color"
        t.interpolation = "Cubic"
        links.new(mapping.outputs["Vector"], t.inputs["Vector"])
        return t

    diff = tex("diffuse.jpg", True)
    tint = nt.new("ShaderNodeMixRGB")
    tint.blend_type = "MULTIPLY"
    tint.inputs["Fac"].default_value = 1.0
    tint.inputs["Color2"].default_value = [*(np.asarray(P["tint"]) * P["brightness"]).clip(0, 2), 1.0]
    links.new(diff.outputs["Color"], tint.inputs["Color1"])
    links.new(tint.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(tex("rough.jpg", False).outputs["Color"], bsdf.inputs["Roughness"])
    nmap = nt.new("ShaderNodeNormalMap")
    links.new(tex("normal_gl.jpg", False).outputs["Color"], nmap.inputs["Color"])
    links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    if P["kind"] == "cloth":
        bsdf.inputs["Sheen Weight"].default_value = 0.5
    obj.replace_materials(mat)


def ceramic_material(obj, c: dict) -> None:
    """Glazed dish: body colour under a clear glossy coat, optional stoneware speckle and a
    coloured rim band (by radius, object space), a touch of translucency for porcelain."""
    g = c.get("glaze") or {}
    mat = bproc.material.create("dish")
    nt, links = mat.nodes, mat.links
    bsdf = mat.get_the_one_node_with_type("BsdfPrincipled")
    col = np.asarray(c["color"], float)
    coord = nt.new("ShaderNodeTexCoord").outputs["Object"]
    colour = [*col, 1.0]
    out = None
    if g.get("speckle"):
        noise = nt.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 900.0
        noise.inputs["Detail"].default_value = 2.0
        links.new(coord, noise.inputs["Vector"])
        ramp = nt.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = 0.62, 0.68
        ramp.color_ramp.elements[0].color = [1, 1, 1, 1]
        ramp.color_ramp.elements[1].color = [*(1 - g["speckle"] * np.array([0.7, 0.72, 0.75])), 1.0]
        links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        mul = nt.new("ShaderNodeMixRGB")
        mul.blend_type, mul.inputs["Fac"].default_value = "MULTIPLY", 1.0
        mul.inputs["Color1"].default_value = colour
        links.new(ramp.outputs["Color"], mul.inputs["Color2"])
        out = mul.outputs["Color"]
    if g.get("rim_color") is not None:
        sep = nt.new("ShaderNodeSeparateXYZ")
        links.new(coord, sep.inputs["Vector"])
        r = nt.new("ShaderNodeVectorMath")
        r.operation = "LENGTH"
        comb = nt.new("ShaderNodeCombineXYZ")
        links.new(sep.outputs["X"], comb.inputs["X"])
        links.new(sep.outputs["Y"], comb.inputs["Y"])
        links.new(comb.outputs["Vector"], r.inputs[0])
        band = nt.new("ShaderNodeMapRange")
        band.interpolation_type = "SMOOTHSTEP"
        R = float(c["R"] + c.get("rim_w", 0.0))
        band.inputs["From Min"].default_value = g["rim_frac"] * R - 0.0015
        band.inputs["From Max"].default_value = g["rim_frac"] * R + 0.0015
        links.new(r.outputs["Value"], band.inputs["Value"])
        mix = nt.new("ShaderNodeMixRGB")
        links.new(band.outputs["Result"], mix.inputs["Fac"])
        if out is not None:
            links.new(out, mix.inputs["Color1"])
        else:
            mix.inputs["Color1"].default_value = colour
        mix.inputs["Color2"].default_value = [*g["rim_color"], 1.0]
        out = mix.outputs["Color"]
    if out is not None:
        links.new(out, bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = colour
    bsdf.inputs["Roughness"].default_value = float(c["roughness"])
    bsdf.inputs["Coat Weight"].default_value = float(g.get("coat", 0.8))
    bsdf.inputs["Coat Roughness"].default_value = float(g.get("coat_rough", 0.05))
    bsdf.inputs["Subsurface Weight"].default_value = float(g.get("sss", 0.0))
    bsdf.inputs["Subsurface Scale"].default_value = 0.002
    obj.replace_materials(mat)


def leaf_uvs(obj, grid: tuple[int, int]) -> None:
    """UVs for the salad sheets (synth.geometry.salad_sheets: per leaf an n_u x n_v vertex grid,
    in order), u along the midrib, v across; cloth and solidify keep them."""
    me = obj.get_mesh()
    n_u, n_v = grid
    per = n_u * n_v
    uv = me.uv_layers.new(name="leaf")
    loops_v = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", loops_v)
    k = loops_v % per
    co = np.stack([(k // n_v) / (n_u - 1), (k % n_v) / (n_v - 1)], 1).astype(np.float32)
    uv.data.foreach_set("uv", co.ravel())


def leaf_material(obj, name: str, color, roughness: float) -> None:
    """A leaf: pale midrib and lateral veins from the UVs, darker / yellowed margin, mottling,
    waxy sheen, and light through the leaf (translucency)."""
    mat = bproc.material.create(name)
    nt, links = mat.nodes, mat.links
    bsdf = mat.get_the_one_node_with_type("BsdfPrincipled")
    out = mat.get_the_one_node_with_type("OutputMaterial")
    c = np.asarray(color, float)
    uv = nt.new("ShaderNodeUVMap")
    uv.uv_map = "leaf"
    sep = nt.new("ShaderNodeSeparateXYZ")
    links.new(uv.outputs["UV"], sep.inputs["Vector"])

    def math(op, a, b=None):
        n = nt.new("ShaderNodeMath")
        n.operation = op
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = float(v)
            else:
                links.new(v, n.inputs[i])
        return n.outputs[0]

    across = math("ABSOLUTE", math("SUBTRACT", sep.outputs["Y"], 0.5))            # 0 at midrib, 0.5 at edge
    mottle = nt.new("ShaderNodeTexNoise")
    mottle.inputs["Scale"].default_value = 6.0
    links.new(uv.outputs["UV"], mottle.inputs["Vector"])
    wob = nt.new("ShaderNodeTexNoise")                                               # irregular veins
    wob.inputs["Scale"].default_value = 3.0
    links.new(uv.outputs["UV"], wob.inputs["Vector"])
    # soft midrib, fading towards the tip
    midrib = math("MULTIPLY", math("SUBTRACT", 1.0, math("MINIMUM", math("MULTIPLY", across, 22.0), 1.0)),
                  math("SUBTRACT", 1.0, math("MULTIPLY", sep.outputs["X"], 0.6)))
    # lateral veins from the midrib towards the margin: lines of u - 0.6 * across = const, wobbly,
    # faint, gone before the edge
    phase = math("ADD", math("SUBTRACT", sep.outputs["X"], math("MULTIPLY", across, 0.6)),
                 math("MULTIPLY", wob.outputs["Fac"], 0.15))
    d = math("ABSOLUTE", math("SUBTRACT", math("FRACT", math("MULTIPLY", phase, 6.0)), 0.5))
    veins = math("MULTIPLY", math("SUBTRACT", 1.0, math("MINIMUM", math("MULTIPLY", d, 14.0), 1.0)),
                 math("SUBTRACT", 1.0, math("MINIMUM", math("MULTIPLY", across, 2.4), 1.0)))
    vein = math("MAXIMUM", math("MULTIPLY", midrib, 0.7), math("MULTIPLY", veins, 0.3))
    edge = math("MULTIPLY", math("MINIMUM", math("MAXIMUM", math("MULTIPLY", math("SUBTRACT", across, 0.38), 6.0), 0.0), 1.0),
                mottle.outputs["Fac"])

    base = nt.new("ShaderNodeMixRGB")
    links.new(mottle.outputs["Fac"], base.inputs["Fac"])
    base.inputs["Color1"].default_value = [*(c * 0.75).clip(0, 1), 1]
    base.inputs["Color2"].default_value = [*(c * 1.15).clip(0, 1), 1]
    edged = nt.new("ShaderNodeMixRGB")
    links.new(edge, edged.inputs["Fac"])
    links.new(base.outputs["Color"], edged.inputs["Color1"])
    edged.inputs["Color2"].default_value = [*(c * np.array([1.1, 1.05, 0.7]) + [0.03, 0.03, 0]).clip(0, 1), 1]
    veined = nt.new("ShaderNodeMixRGB")
    links.new(vein, veined.inputs["Fac"])
    links.new(edged.outputs["Color"], veined.inputs["Color1"])
    veined.inputs["Color2"].default_value = [*(c * 0.7 + 0.22).clip(0, 1), 1]
    links.new(veined.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = float(roughness)
    bsdf.inputs["Coat Weight"].default_value = 0.15                                   # waxy cuticle
    bump = nt.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    links.new(vein, bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    trans = nt.new("ShaderNodeBsdfTranslucent")
    links.new(veined.outputs["Color"], trans.inputs["Color"])
    mix = nt.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = 0.3
    links.new(bsdf.outputs["BSDF"], mix.inputs[1])
    links.new(trans.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], out.inputs["Surface"])
    obj.replace_materials(mat)


def _ellipsoid(n_u: int = 10, n_v: int = 6):
    th, ph = np.linspace(0, 2 * np.pi, n_u, endpoint=False), np.linspace(0, np.pi, n_v + 1)[1:-1]
    V = [[0, 0, 1]] + [[np.sin(p) * np.cos(t), np.sin(p) * np.sin(t), np.cos(p)] for p in ph for t in th] + [[0, 0, -1]]
    F = [[0, 1 + j, 1 + (j + 1) % n_u] for j in range(n_u)]
    for i in range(len(ph) - 1):
        a, b = 1 + i * n_u, 1 + (i + 1) * n_u
        F += [[a + j, b + j, b + (j + 1) % n_u, a + (j + 1) % n_u] for j in range(n_u)]
    last = len(V) - 1
    a = 1 + (len(ph) - 1) * n_u
    F += [[a + j, last, a + (j + 1) % n_u] for j in range(n_u)]
    return np.asarray(V, float), F


def piece_mesh(shape: str, size_mm, rng: np.random.Generator):
    """One small food piece (unit-free, scaled to size_mm L x W x H), lying in its local XY plane."""
    L, W, H = np.asarray(size_mm, float) / 2000.0          # half extents in metres
    if shape == "cube":
        V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        V += rng.normal(0, 0.12, V.shape)                    # hand-diced
        F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
        return V * [L, W, H], F
    if shape in ("noodle", "penne"):                          # a tube along a wavy (noodle) or straight path
        n_a, n_r = (28, 6) if shape == "noodle" else (6, 10)
        x = np.linspace(-L, L, n_a)
        if shape == "noodle":
            y = 0.25 * L * np.sin(rng.uniform(1.5, 3.5) * np.pi * x / L + rng.uniform(0, 6))
            z = 0.06 * L * np.sin(rng.uniform(1, 2) * np.pi * x / L + rng.uniform(0, 6))
        else:
            y, z = np.zeros(n_a), np.zeros(n_a)
        path = np.stack([x, y, z], 1)
        tang = np.gradient(path, axis=0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        nrm = np.cross(tang, [0.0, 0.0, 1.0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        bin_ = np.cross(tang, nrm)
        r = H
        V = [path[i] + r * (np.cos(b) * nrm[i] + np.sin(b) * bin_[i])
             for i in range(n_a) for b in 2 * np.pi * np.arange(n_r) / n_r]
        if shape == "penne":                                  # angled cut ends
            V = np.asarray(V)
            V[:, 0] += np.where(np.abs(V[:, 0]) > 0.9 * L, 0.6 * r * np.sign(V[:, 0]) * V[:, 2] / r, 0.0)
        F = [[i * n_r + j, (i + 1) * n_r + j, (i + 1) * n_r + (j + 1) % n_r, i * n_r + (j + 1) % n_r]
             for i in range(n_a - 1) for j in range(n_r)]
        return np.asarray(V, float), F
    if shape == "cluster":                                    # granola: a lumpy clump
        V, F = _ellipsoid(12, 8)
        V = V * (1 + rng.normal(0, 0.22, (len(V), 1))) * [L, W, H]
        return V, F
    if shape == "chunk":                                      # a rounded piece of meat / vegetable
        V, F = _ellipsoid(12, 8)
        V = V * (1 + rng.normal(0, 0.08, (len(V), 1))) * [L, W, H]
        return V, F
    if shape == "macaroni":                                   # a curved tube (elbow), open ends
        n_a, n_r = 10, 8
        R, r = L * 0.9, H
        V, F = [], []
        for i in range(n_a):
            a = -1.0 + 2.0 * i / (n_a - 1)                    # arc angle, radians
            ctr = np.array([R * np.sin(a), R * (1 - np.cos(a)) - 0.4 * R, 0.0])
            nrm = np.array([-np.sin(a), np.cos(a), 0.0])
            for j in range(n_r):
                b = 2 * np.pi * j / n_r
                V.append(ctr + r * (np.cos(b) * nrm + np.sin(b) * np.array([0.0, 0.0, 1.0])))
        for i in range(n_a - 1):
            for j in range(n_r):
                F.append([i * n_r + j, (i + 1) * n_r + j, (i + 1) * n_r + (j + 1) % n_r, i * n_r + (j + 1) % n_r])
        return np.asarray(V, float), F
    V, F = _ellipsoid()
    V = V * [L, W, H]
    if shape == "rice":                                       # taper the ends
        V[:, 1:] *= np.sqrt(np.clip(1 - (V[:, :1] / L) ** 2, 0.15, 1)) ** 0.3
    elif shape == "bean":                                     # kidney: bend
        V[:, 1] += 0.35 * W * (V[:, 0] / L) ** 2
    elif shape in ("flake", "lentil", "quinoa"):              # irregular outline
        V[:, :2] *= 1 + rng.normal(0, 0.08, (len(V), 1))
    return V, F


def scatter_pieces(obj, it: dict, seed: int) -> int:
    """Real grains / flakes / beans (synth.looks.PIECES) scattered over the upward-facing surface
    of a settled food with geometry nodes and realised into its mesh, so they carry the item's
    label in every pass. Pieces are sunk part-way into the surface and randomly turned.
    Returns the number of piece kinds added."""
    specs = it.get("pieces") or []
    if not specs:
        return 0
    rng = np.random.default_rng(seed)
    tree = bpy.data.node_groups.new(f"pieces_{it['id']}", "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nt, links = tree.nodes, tree.links
    gin, gout = nt.new("NodeGroupInput"), nt.new("NodeGroupOutput")
    join = nt.new("GeometryNodeJoinGeometry")
    links.new(gin.outputs[0], join.inputs[0])
    normal = nt.new("GeometryNodeInputNormal")
    nz = nt.new("ShaderNodeSeparateXYZ")
    links.new(normal.outputs[0], nz.inputs[0])
    up = nt.new("FunctionNodeCompare")
    up.data_type, up.operation = "FLOAT", "GREATER_THAN"
    links.new(nz.outputs["Z"], up.inputs[0])
    up.inputs[1].default_value = 0.3
    for k, (shape, density, size, colour, sink) in enumerate(specs):
        V, F = piece_mesh(shape, size, rng)
        me = bpy.data.meshes.new(f"piece_{it['id']}_{k}")
        me.from_pydata(V.tolist(), [], F)
        me.validate()
        for p in me.polygons:
            p.use_smooth = True
        src = bpy.data.objects.new(f"piece_{it['id']}_{k}", me)
        bpy.context.scene.collection.objects.link(src)
        src.hide_render = True
        src.hide_viewport = True
        # pieces of the food itself (rice grains, flakes, macaroni) are a little paler, glossier and
        # more translucent than the paste between them, so each grain reads against its neighbours
        own = colour is None
        rgb = np.clip(np.asarray(it.get("color") or [0.7, 0.6, 0.4]) * 1.12, 0, 1) if own else np.asarray(colour)
        m = bpy.data.materials.new(f"piece_{it['id']}_{k}")
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = [*rgb, 1.0]
        b.inputs["Roughness"].default_value = 0.35 if own else 0.45
        b.inputs["Subsurface Weight"].default_value = 0.5 if own else 0.2
        b.inputs["Subsurface Radius"].default_value = (1.0, 0.8, 0.6)
        b.inputs["Subsurface Scale"].default_value = 0.0015
        b.inputs["Coat Weight"].default_value = 0.15 if own else 0.0
        me.materials.append(m)
        dist = nt.new("GeometryNodeDistributePointsOnFaces")
        dist.distribute_method = "RANDOM"
        links.new(gin.outputs[0], dist.inputs["Mesh"])
        links.new(up.outputs[0], dist.inputs["Selection"])
        dist.inputs["Density"].default_value = float(density)
        dist.inputs["Seed"].default_value = int(rng.integers(1 << 20))
        info = nt.new("GeometryNodeObjectInfo")
        info.inputs["Object"].default_value = src
        info.inputs["As Instance"].default_value = True
        inst = nt.new("GeometryNodeInstanceOnPoints")
        links.new(dist.outputs["Points"], inst.inputs["Points"])
        links.new(info.outputs["Geometry"], inst.inputs["Instance"])
        links.new(dist.outputs["Rotation"], inst.inputs["Rotation"])           # local Z = surface normal
        sc = nt.new("FunctionNodeRandomValue")
        sc.data_type = "FLOAT"
        sc.inputs[2].default_value, sc.inputs[3].default_value = 0.8, 1.2           # Min, Max (float)
        sc.inputs["Seed"].default_value = int(rng.integers(1 << 20))
        links.new(sc.outputs[1], inst.inputs["Scale"])
        rot = nt.new("GeometryNodeRotateInstances")
        rv = nt.new("FunctionNodeRandomValue")
        rv.data_type = "FLOAT_VECTOR"
        rv.inputs[0].default_value = (-0.35, -0.35, 0.0)
        rv.inputs[1].default_value = (0.35, 0.35, 2 * np.pi)
        rv.inputs["Seed"].default_value = int(rng.integers(1 << 20))
        links.new(inst.outputs["Instances"], rot.inputs["Instances"])
        links.new(rv.outputs[0], rot.inputs["Rotation"])
        rot.inputs["Local Space"].default_value = True
        tr = nt.new("GeometryNodeTranslateInstances")
        links.new(rot.outputs["Instances"], tr.inputs["Instances"])
        tr.inputs["Translation"].default_value = (0.0, 0.0, -float(sink) * size[2] / 1000.0)
        tr.inputs["Local Space"].default_value = True
        links.new(tr.outputs["Instances"], join.inputs[0])
    real = nt.new("GeometryNodeRealizeInstances")
    links.new(join.outputs[0], real.inputs[0])
    links.new(real.outputs[0], gout.inputs[0])
    mod = obj.blender_obj.modifiers.new("pieces", "NODES")
    mod.node_group = tree
    return len(specs)


def collider_proxy(obj, name: str, max_faces: int = 20000):
    """A decimated copy of a big static scan (scanned dish) for Bullet's mesh collider: a
    full-resolution GSO dish stalled the rigid-body drop for ~15 min. Returns None if small."""
    n = len(obj.get_mesh().polygons)
    if n <= max_faces:
        return None
    proxy = bproc.object.create_with_empty_mesh(name)
    proxy.blender_obj.data = obj.blender_obj.data.copy()
    proxy.blender_obj.matrix_world = obj.blender_obj.matrix_world.copy()
    dec = proxy.blender_obj.modifiers.new("decimate", "DECIMATE")
    dec.ratio = max_faces / n
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(proxy.blender_obj.evaluated_get(dg))
    proxy.blender_obj.modifiers.clear()
    proxy.blender_obj.data = me
    proxy.hide()
    return proxy


def surface_samples(obj, rng: np.random.Generator, per_m2: float = 1e6, lo: int = 500,
                    hi: int = 60000) -> np.ndarray:
    """Area-uniform points on the object's surface in world coordinates (~1 per mm^2)."""
    me = obj.blender_obj.data
    me.calc_loop_triangles()
    tri = np.empty(len(me.loop_triangles) * 3, np.int64)
    me.loop_triangles.foreach_get("vertices", tri)
    co = np.empty(len(me.vertices) * 3, np.float64)
    me.vertices.foreach_get("co", co)
    M = np.array(obj.blender_obj.matrix_world)
    V = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    T = V[tri.reshape(-1, 3)]
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    n = int(np.clip(area.sum() * per_m2, lo, hi))
    k = rng.choice(len(T), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = T[k]
    return (t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])).astype(np.float32)


def hull_proxy(obj, name: str, max_pts: int = 20000, min_verts: int = 5000):
    """Convex hull of (a random subset of) obj's vertices as a separate object with the same
    transform, or None when obj is already small enough to simulate directly."""
    me = obj.get_mesh()
    if len(me.vertices) <= min_verts:
        return None
    co = np.empty(len(me.vertices) * 3, np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    if len(co) > max_pts:
        co = co[np.random.default_rng(0).choice(len(co), max_pts, replace=False)]
    bm = bmesh.new()
    for p in co:
        bm.verts.new(p)
    res = bmesh.ops.convex_hull(bm, input=bm.verts)
    loose = {g for g in res["geom_interior"] + res["geom_unused"] if isinstance(g, bmesh.types.BMVert)}
    bmesh.ops.delete(bm, geom=list(loose), context="VERTS")
    proxy = bproc.object.create_with_empty_mesh(name)
    bm.to_mesh(proxy.get_mesh())
    bm.free()
    copy_pose(obj, proxy)
    return proxy


def copy_pose(src, dst) -> None:
    """Location / rotation / scale (matrix_world can be stale until the depsgraph updates)."""
    d, s = dst.blender_obj, src.blender_obj
    d.rotation_mode = s.rotation_mode
    d.location, d.rotation_euler, d.scale = s.location.copy(), s.rotation_euler.copy(), s.scale.copy()


def world_volume_ml(obj) -> float:
    bm = bmesh.new()
    bm.from_mesh(obj.get_mesh())
    bm.transform(obj.blender_obj.matrix_world)
    v = bm.calc_volume(signed=False)
    bm.free()
    return float(v * 1e6)


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("out")
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--no-physics", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="render on CPU (when the GPU is full)")
    args = ap.parse_args(argv)

    t0 = time.time()
    spec_path = Path(args.spec)
    scene_dir = spec_path.parent
    spec = json.loads(spec_path.read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bproc.init()
    # BlenderProc defaults to 0 glossy / transmission bounces: no reflections of the room or of
    # neighbouring food in glossy sauces and glazes, and no light through leaves
    bproc.renderer.set_light_bounces(diffuse_bounces=4, glossy_bounces=4, transmission_bounces=4,
                                     max_bounces=8, transparent_max_bounces=8)

    # table
    tab = spec["table"]
    table = bproc.object.create_primitive("PLANE", scale=[tab["size_m"] / 2] * 3)
    if tab.get("pbr"):                        # photographed wood / stone / laminate / cloth
        pbr_table(table, tab)
    elif tab.get("texture_scale"):            # mottled wood / laminate / cloth instead of flat colour
        textured_material(table, "table", tab["color"], tab["roughness"], tab["texture_scale"])
    else:
        flat_material(table, "table", tab["color"], tab["roughness"])
    table.enable_rigidbody(False, collision_shape="BOX", collision_margin=0.0005)

    # container (static)
    c = spec["container"]
    if c.get("asset"):                    # a real scanned plate / bowl (GSO / YCB), centred on the axis
        cont = load_mesh(c["asset"], scene_dir)
        cont.persist_transformation_into_mesh()
        place_on_table(cont, [0, 0, 0], [0, 0, c["asset"].get("yaw", 0.0)])
        bb = np.array(cont.get_bound_box())
        loc = np.array(cont.get_location())
        cont.set_location([loc[0] - bb[:, 0].mean(), loc[1] - bb[:, 1].mean(), loc[2]])
    else:
        cont = mesh_from_npz(scene_dir / c["mesh"], "container")
        for poly in cont.get_mesh().polygons:
            poly.use_smooth = True
        ceramic_material(cont, c)
    cont.set_cp("category_id", CONTAINER_ID)
    cont_collider = collider_proxy(cont, "container_collider") if c.get("asset") else None
    (cont_collider or cont).enable_rigidbody(False, collision_shape="MESH", collision_margin=0.0005)

    # food
    objs, proxies, part_objs = [], [], {}
    for it in spec["items"]:
        o = load_mesh(it, scene_dir)
        o.persist_transformation_into_mesh()       # importer axis conversion -> mesh (cached parts' frame)
        s = it["to_m"] * it["scale"]
        o.set_scale([s, s, s])
        if it.get("color") is not None or it["fmt"] == "npz":
            col = it.get("color") or [0.6, 0.45, 0.3]
            # brightness varies batch to batch; hue only a little (per-channel +-15 % turned mash green)
            jitter = np.clip(np.asarray(col) * np.random.uniform(0.88, 1.12) * np.random.normal(1, 0.03, 3), 0, 1)
            if it.get("cloth"):
                leaf_uvs(o, tuple(it["cloth"].get("grid", (12, 5))))
                leaf_material(o, f"food_{it['id']}", jitter.tolist(), it.get("roughness", 0.5))
            elif it.get("look"):
                food_material(o, f"food_{it['id']}", jitter.tolist(), it.get("roughness", 0.5), it["look"])
            elif it.get("texture"):
                scale = 80 if it["source"] == "procedural_topping" else np.random.uniform(150, 400)
                textured_material(o, f"food_{it['id']}", jitter.tolist(), it.get("roughness", 0.5), scale)
            else:
                flat_material(o, f"food_{it['id']}", jitter.tolist(), it.get("roughness", 0.5))
        o.set_cp("category_id", it["id"])
        proxy = None
        if it.get("matrix_world") is not None:     # settled by synth.mpm with the soft food
            o.blender_obj.matrix_world = Matrix(it["matrix_world"])
            if it.get("collision"):
                part_objs[it["id"]] = compound_collider(o, collision_parts(it["collision"]), np.zeros(3), None)
        elif it["dynamic"]:
            # init_location is the centre, not the base; mesh coords move by -c_local
            org = np.array(o.set_origin(mode="CENTER_OF_VOLUME"))
            c_local = org / s
            o.set_location(it["init_location"])
            o.set_rotation_euler(it["init_euler"])
            mass = (it["mass_g"] or it["volume_ml"]) / 1000.0      # kg; ~1 g/mL when unknown
            damping = it.get("damping", 0.3)
            if it.get("collision"):          # CoACD parts (synth.collision): true-shape contacts
                part_objs[it["id"]] = compound_collider(o, collision_parts(it["collision"]), c_local, mass, damping)
            else:
                # scans have up to ~1 M vertices and Bullet's hull shape walks all of them per
                # contact query, so simulate a small hull of the same shape and copy its pose back
                proxy = hull_proxy(o, f"proxy_{it['id']}")
                (proxy or o).enable_rigidbody(True, collision_shape="CONVEX_HULL", mass=mass, friction=0.8,
                                              linear_damping=damping, angular_damping=damping,
                                              collision_margin=0.0005)
        elif not it.get("cloth") and not it.get("mpm"):   # cloth (leaves): simulated after the rigid bodies
            o.enable_rigidbody(False, collision_shape="MESH", collision_margin=0.0005)
        objs.append((it, o))
        if proxy:
            proxies.append((o, proxy))
    print(f"[render] scene loaded in {time.time() - t0:.0f}s ({len(proxies)} physics proxies)", flush=True)

    # non-food clutter on the table (hard preset): static, its own label
    clutter = []
    for i, d in enumerate(spec.get("distractors", [])):
        if d.get("fmt"):                  # a real scanned prop (YCB / GSO): its own textures
            o = load_mesh(d, scene_dir)
            o.persist_transformation_into_mesh()
            place_on_table(o, d["location"], d["euler"])
        else:
            o = mesh_from_npz(scene_dir / d["mesh"], f"distractor_{i}")
            o.set_location(d["location"])
            o.set_rotation_euler(d["euler"])
            textured_material(o, f"distractor_{i}", d["color"], d["roughness"], scale=40, metallic=d["metallic"])
        o.set_cp("category_id", DISTRACTOR_ID)
        o.enable_rigidbody(False, collision_shape="MESH", collision_margin=0.0005)
        clutter.append(o)

    if not args.no_physics and any(it["dynamic"] for it, _ in objs):
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=1.5, max_simulation_time=6, check_object_interval=0.5,
            substeps_per_frame=40, solver_iters=30)
        print(f"[render] physics done at {time.time() - t0:.0f}s", flush=True)
    for o, proxy in proxies:
        copy_pose(proxy, o)
        proxy.delete()
    bpy.context.view_layer.update()

    cloth = [(it, o) for it, o in objs if it.get("cloth")]
    if cloth:
        # obstacles: table, dish, static food, and for dropped scans their convex parts (the
        # full-resolution scan would make the cloth collision BVH huge)
        obstacles = [table, cont] + [o for it, o in objs if not it.get("cloth") and it["id"] not in part_objs]
        obstacles += [k for kids in part_objs.values() for k in kids]
        simulate_cloth(cloth, obstacles, keep_r=0.95 * (c["R"] + c.get("rim_w", 0.0)))
        print(f"[render] cloth done at {time.time() - t0:.0f}s", flush=True)

    # real grains / flakes / beans on the settled soft food (after cloth: not leaf obstacles)
    n_kinds = sum(scatter_pieces(o, it, seed=1000 + it["id"]) for it, o in objs if it.get("pieces"))
    if n_kinds:
        print(f"[render] scattered {n_kinds} piece kinds", flush=True)

    # lighting
    L = spec["lighting"]
    if L.get("hdri"):                       # a real room: window light, bounce, reflections
        bproc.world.set_world_background_hdr_img(L["hdri"], strength=L["hdri_strength"],
                                                 rotation_euler=[0.0, 0.0, L.get("hdri_rotation", 0.0)])
    else:
        tint = np.asarray(L.get("ambient_tint", [1.0, 1.0, 1.0]))     # warm / cool room light
        bproc.renderer.set_world_background((L["ambient"] * tint).tolist(), strength=1.0)
    if L.get("key"):                        # with an HDRI: an optional soft window / lamp on top
        key = bproc.types.Light()
        key.set_type("AREA")
        key.set_location(L["key"]["location"])
        key.set_rotation_mat(bproc.camera.rotation_from_forward_vec(-np.asarray(L["key"]["location"])))
        key.set_energy(L["key"]["energy_w"])
        key.set_color(L["key"].get("color", [1.0, 1.0, 1.0]))
        key.blender_obj.data.size = L["key"]["size_m"]

    # colour management: Blender's default AgX desaturates bright colours (chili rendered pink);
    # a phone pipeline is closer to a plain sRGB curve, with auto-exposure that varies per scene
    look = spec.get("look", {})
    vs = bpy.context.scene.view_settings
    vs.view_transform = look.get("view_transform", vs.view_transform)
    vs.look = look.get("look", "None")
    vs.exposure = float(look.get("exposure", 0.0))
    if look.get("hdr"):           # scene-linear EXR; synth.camera.develop does exposure + tone curve
        bproc.renderer.set_output_format(file_format="OPEN_EXR", color_depth=16)

    # camera: ARKit K has pixel centres at +0.5, BlenderProc at integers
    cam = spec["camera"]
    K = np.array(cam["K"])
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    bproc.camera.set_intrinsics_from_K_matrix(K, cam["width"], cam["height"], clip_start=0.01, clip_end=20.0)
    for T in cam["c2w"]:
        bproc.camera.add_camera_pose(np.array(T))
    if cam.get("dof"):                      # focus on the dish; a phone's deep but finite depth of field
        focus = bproc.object.create_empty("focus")
        focus.set_location(cam["dof"]["focus"])
        bproc.camera.add_depth_of_field(focus, cam["dof"]["fstop"])

    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(map_by=["category_id"], default_values={"category_id": 0})
    bproc.renderer.set_max_amount_of_samples(args.spp)
    if args.cpu:
        bproc.renderer.set_render_devices(use_only_cpu=True)
    t_setup = time.time()
    data = bproc.renderer.render()
    t_render = time.time()

    for i, (rgb, depth, seg) in enumerate(zip(data["colors"], data["depth"], data["category_id_segmaps"])):
        depth = np.asarray(depth, np.float32)
        depth[~np.isfinite(depth) | (depth > 1e4)] = 0.0
        rgb = np.asarray(rgb)[..., :3]
        col = {"rgb_linear": rgb.astype(np.float16)} if look.get("hdr") else {"rgb": rgb.astype(np.uint8)}
        np.savez_compressed(out / f"frame_{i:06d}.npz", **col, depth=depth, seg=np.asarray(seg).astype(np.uint8))

    R = cont.blender_obj.matrix_world
    items = []
    for it, o in objs:
        M = np.array(o.get_local2world_mat())
        bb = np.array(o.get_bound_box())
        ctr = bb.mean(0)
        items.append({"id": it["id"], "name": it["name"], "local2world": M.tolist(),
                      "bbox_world": [bb.min(0).tolist(), bb.max(0).tolist()],
                      "scene_volume_ml": world_volume_ml(o), "spec_volume_ml": it["volume_ml"],
                      "radius_from_axis_m": float(np.hypot(ctr[0], ctr[1])),
                      "min_z_m": float(bb[:, 2].min())})
    # surface samples of every labelled object in its final pose: per-point 3D ground truth
    # for the fused cloud (finalize moves them to the ARKit frame)
    srng = np.random.default_rng(0)
    pts, lab = [], []
    for lid, o in [(it["id"], o) for it, o in objs] + [(CONTAINER_ID, cont)] + [(DISTRACTOR_ID, o) for o in clutter]:
        p = surface_samples(o, srng)
        pts.append(p)
        lab.append(np.full(len(p), lid, np.uint8))
    np.savez_compressed(out / "gt_points.npz", xyz=np.concatenate(pts), label=np.concatenate(lab))

    meta = {"scene_id": spec["scene_id"], "spp": args.spp, "n_frames": len(data["colors"]),
            "container_local2world": [list(r) for r in R], "items": items,
            "timing_s": {"setup": t_setup - t0, "render": t_render - t_setup, "total": time.time() - t0}}
    (out / "render_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[render] {spec['scene_id']}: {meta['n_frames']} frames, {meta['timing_s']['render']:.1f}s render")


main()
