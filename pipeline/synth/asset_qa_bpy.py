"""Blender helper for synth.asset_qa: one small 3/4-view workbench thumbnail (true texture colours, flat light)
per asset.

    blender -b --python asset_qa_bpy.py -- jobs.json OUT_DIR [size]

jobs.json: [{id, path, fmt, color}]; writes OUT_DIR/<id>.jpg (skips existing). Not importable outside Blender.
"""
import json
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
jobs = json.load(open(argv[0]))
out = argv[1]
S = int(argv[2]) if len(argv) > 2 else 192
os.makedirs(out, exist_ok=True)

sc = bpy.context.scene
sc.render.engine = "BLENDER_WORKBENCH"
sc.render.resolution_x = sc.render.resolution_y = S
sc.render.image_settings.file_format = "JPEG"
sc.render.image_settings.quality = 80
sh = sc.display.shading
sh.light = "FLAT"
sh.show_object_outline = False
sc.view_settings.view_transform = "Standard"
if sc.world is None:
    sc.world = bpy.data.worlds.new("w")
sc.world.color = (0.55, 0.55, 0.58)
cam_d = bpy.data.cameras.new("c")
cam_d.type = "ORTHO"
cam = bpy.data.objects.new("c", cam_d)
sc.collection.objects.link(cam)
sc.camera = cam


def clear():
    for o in list(bpy.data.objects):
        if o is not cam:
            bpy.data.objects.remove(o, do_unlink=True)
    for blk in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for b in list(blk):
            if b.users == 0:
                blk.remove(b)


def load(j):
    p, fmt = j["path"], j["fmt"]
    if fmt == "npz":
        d = np.load(p)
        me = bpy.data.meshes.new("m")
        me.from_pydata(d["V"].tolist(), [], d["F"].tolist())
        ob = bpy.data.objects.new("m", me)
        sc.collection.objects.link(ob)
        mat = bpy.data.materials.new("c")
        c = j.get("color") or [0.5, 0.5, 0.5]
        mat.diffuse_color = tuple(float(x) ** 0.45 for x in c) + (1.0,)
        me.materials.append(mat)
        sh.color_type = "MATERIAL"
        return [ob]
    sh.color_type = "TEXTURE"
    if fmt == "obj":
        bpy.ops.wm.obj_import(filepath=p, forward_axis="Y", up_axis="Z")   # keep the file's own axes
    elif fmt in ("gltf", "glb"):
        bpy.ops.import_scene.gltf(filepath=p)
    elif fmt == "ply":
        bpy.ops.wm.ply_import(filepath=p)
    return [o for o in bpy.context.selected_objects if o.type == "MESH"]


for j in jobs:
    fn = os.path.join(out, j["id"] + ".jpg")
    if os.path.exists(fn):
        continue
    t0 = time.time()
    try:
        clear()
        obs = load(j)
        if not obs:
            raise RuntimeError("no mesh")
        P = np.array([[v.x, v.y, v.z] for o in obs for v in (o.matrix_world @ Vector(c) for c in o.bound_box)])
        lo, hi = P.min(0), P.max(0)
        ctr, R = (lo + hi) / 2, np.linalg.norm(hi - lo) / 2
        d = Vector((1, -1.1, 0.85)).normalized()
        cam.location = Vector(ctr) + d * (R * 4 + 1)
        cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        cam_d.ortho_scale = R * 2.15
        cam_d.clip_start, cam_d.clip_end = 0.001, R * 20 + 10
        sc.render.filepath = fn
        bpy.ops.render.render(write_still=True)
    except Exception as e:  # noqa: BLE001
        print("FAIL", j["id"], e)
        open(fn + ".fail", "w").write(str(e))
    print("done", j["id"], round(time.time() - t0, 1), flush=True)
