import os
import json
import shutil

def create_result_dir(path, args=None, overwrite=True):
    if overwrite and os.path.exists(path):
        shutil.rmtree(path)
    assert not os.path.exists(path), "Out path exists, choose different dir"

    os.makedirs(path)
    if args is not None:
        with open(os.path.join(path, 'args.json'), 'w') as f:
            f.write(json.dumps(vars(args), indent=4))
