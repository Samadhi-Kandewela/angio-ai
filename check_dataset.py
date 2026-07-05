import json, os

base = "dataset"

for split in ["train", "val"]:
    for ds in ["syntax", "stenosis"]:
        path = os.path.join(base, ds, split, "annotations", f"{split}.json")
        if not os.path.exists(path):
            print(f"{ds}/{split}: NOT FOUND")
            continue
        with open(path) as f:
            data = json.load(f)
        print(f"{ds}/{split}: {len(data['images'])} images, {len(data['annotations'])} annotations")

print()
for split in ["train", "val"]:
    sp = os.path.join(base, "syntax", split, "annotations", f"{split}.json")
    st = os.path.join(base, "stenosis", split, "annotations", f"{split}.json")
    if not os.path.exists(sp) or not os.path.exists(st):
        continue
    with open(sp) as f:
        syntax_imgs = {img["file_name"] for img in json.load(f)["images"]}
    with open(st) as f:
        stenosis_imgs = {img["file_name"] for img in json.load(f)["images"]}

    both = syntax_imgs & stenosis_imgs
    syntax_only = syntax_imgs - stenosis_imgs
    stenosis_only = stenosis_imgs - syntax_imgs
    print(f"{split} overlap:")
    print(f"  In both:       {len(both)}")
    print(f"  Syntax only:   {len(syntax_only)}  <- currently excluded from training!")
    print(f"  Stenosis only: {len(stenosis_only)} <- have no anatomy labels")
