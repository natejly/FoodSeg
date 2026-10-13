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


def load_mesh(item: dict, scene_dir: Path):
    p = Path(item["mesh"])
    if not p.is_absolute():
        p = scene_dir / p
    if item["fmt"] == "npz":
        return mesh_from_npz(p, item["name"])
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

    # table
    tab = spec["table"]
    table = bproc.object.create_primitive("PLANE", scale=[tab["size_m"] / 2] * 3)
    flat_material(table, "table", tab["color"], tab["roughness"])
    table.enable_rigidbody(False, collision_shape="BOX", collision_margin=0.0005)

    # container (static)
    c = spec["container"]
    cont = mesh_from_npz(scene_dir / c["mesh"], "container")
    flat_material(cont, "dish", c["color"], c["roughness"])
    cont.set_cp("category_id", CONTAINER_ID)
    cont.enable_rigidbody(False, collision_shape="MESH", collision_margin=0.0005)

    # food
    objs, proxies = [], []
    for it in spec["items"]:
        o = load_mesh(it, scene_dir)
        s = it["to_m"] * it["scale"]
        o.set_scale([s, s, s])
        if it.get("color") is not None or it["fmt"] == "npz":
            col = it.get("color") or [0.6, 0.45, 0.3]
            jitter = np.clip(np.asarray(col) * np.random.uniform(0.85, 1.15, 3), 0, 1)
            if it.get("texture"):
                scale = 80 if it["source"] == "procedural_topping" else np.random.uniform(150, 400)
                textured_material(o, f"food_{it['id']}", jitter.tolist(), it.get("roughness", 0.5), scale)
            else:
                flat_material(o, f"food_{it['id']}", jitter.tolist(), it.get("roughness", 0.5))
        o.set_cp("category_id", it["id"])
        proxy = None
        if it["dynamic"]:
            o.set_origin(mode="CENTER_OF_VOLUME")   # init_location is the centre, not the base
            o.set_location(it["init_location"])
            o.set_rotation_euler(it["init_euler"])
            mass = (it["mass_g"] or it["volume_ml"]) / 1000.0      # kg; ~1 g/mL when unknown
            # scans have up to ~1 M vertices and Bullet's hull shape walks all of them per
            # contact query, so simulate a small hull of the same shape and copy its pose back
            proxy = hull_proxy(o, f"proxy_{it['id']}")
            (proxy or o).enable_rigidbody(True, collision_shape="CONVEX_HULL", mass=mass, friction=0.8,
                                          linear_damping=it.get("damping", 0.3),
                                          angular_damping=it.get("damping", 0.3), collision_margin=0.0005)
        else:
            o.enable_rigidbody(False, collision_shape="MESH", collision_margin=0.0005)
        objs.append((it, o))
        if proxy:
            proxies.append((o, proxy))
    print(f"[render] scene loaded in {time.time() - t0:.0f}s ({len(proxies)} physics proxies)", flush=True)

    # non-food clutter on the table (hard preset): static, its own label
    clutter = []
    for i, d in enumerate(spec.get("distractors", [])):
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

    # lighting
    L = spec["lighting"]
    if L.get("hdri"):
        bproc.world.set_world_background_hdr_img(L["hdri"], strength=L["hdri_strength"])
    else:
        bproc.renderer.set_world_background([L["ambient"]] * 3, strength=1.0)
    key = bproc.types.Light()
    key.set_type("AREA")
    key.set_location(L["key"]["location"])
    key.set_rotation_mat(bproc.camera.rotation_from_forward_vec(-np.asarray(L["key"]["location"])))
    key.set_energy(L["key"]["energy_w"])
    key.blender_obj.data.size = L["key"]["size_m"]

    # camera: ARKit K has pixel centres at +0.5, BlenderProc at integers
    cam = spec["camera"]
    K = np.array(cam["K"])
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    bproc.camera.set_intrinsics_from_K_matrix(K, cam["width"], cam["height"], clip_start=0.01, clip_end=20.0)
    for T in cam["c2w"]:
        bproc.camera.add_camera_pose(np.array(T))

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
        np.savez_compressed(out / f"frame_{i:06d}.npz", rgb=np.asarray(rgb, np.uint8)[..., :3],
                            depth=depth, seg=np.asarray(seg).astype(np.uint8))

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
