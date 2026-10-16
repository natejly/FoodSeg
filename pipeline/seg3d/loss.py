import torch
import torch.nn.functional as F


def seg_loss(out, batch, cls_w, off_w=1.0, dir_w=0.1, mass_w=0.0, model=None, koc=None):
    dev = out["sem"].device
    w = batch["weight"].to(dev)
    sem = batch["sem"].to(dev)
    ce = F.cross_entropy(out["sem"].float(), sem, weight=torch.tensor(cls_w, device=dev), reduction="none")
    l_sem = (ce * w).sum() / w.sum().clamp(min=1)
    food = (sem == 3) & (w > 0)
    tgt = batch["offset"].to(dev)
    if food.any():
        pr = out["offset"].float()[food]
        t = tgt[food]
        l_off = F.smooth_l1_loss(pr, t, beta=0.1)
        cos = F.cosine_similarity(pr, t, dim=1, eps=1e-6)
        l_dir = (1 - cos)[t.norm(dim=1) > 0.05].mean() if (t.norm(dim=1) > 0.05).any() else pr.sum() * 0
    else:
        l_off = l_dir = out["offset"].sum() * 0
    loss = l_sem + off_w * l_off + dir_w * l_dir
    parts = dict(sem=float(l_sem.detach()), off=float(l_off.detach()), dir=float(l_dir.detach()))
    if koc is not None and "koc" in out:
        from .koc import koc_loss
        lk, pk = koc_loss(out["koc"], batch, koc.get("eps", 0.05), koc.get("change_w", 2.0), koc.get("gamma", 0.0))
        loss = loss + koc.get("w", 1.0) * lk
        parts.update(pk)
    if mass_w > 0 and model is not None and getattr(model.heads, "mass", None) is not None:
        lm, cnt, start = out["feat"].sum() * 0, 0, 0
        for b, items in enumerate(batch["items"]):
            sl = slice(start, start + int(batch["n"][b]))
            start += int(batch["n"][b])
            inst = batch["inst"][sl].to(dev)
            ids = [int(i) for i in inst.unique() if i > 0 and items.get(int(i), {}).get("mass_g")
                   and items[int(i)].get("volume_ml")]
            if ids:
                lv = torch.log(torch.tensor([items[i]["volume_ml"] for i in ids], device=dev))
                tm = torch.log(torch.tensor([float(items[i]["mass_g"]) for i in ids], device=dev))
                lm = lm + F.l1_loss(model.heads.mass_pred(out["feat"][sl], inst, ids, lv).float(), tm, reduction="sum")
                cnt += len(ids)
        if cnt:
            loss = loss + mass_w * lm / cnt
            parts["mass"] = float(lm / cnt)
    return loss, parts
