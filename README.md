# Multi2vec Bind module

🎯 Overview
-----------

This is a multi modal inference container it uses Meta's open source [ImageBind](https://github.com/facebookresearch/ImageBind) implementation as base for this module.

📦 Requirements
----------------

The best way to start working with it would be to first to create a virtual env, activate it and adjust `PYTHONPATH` environment variable to have the modules to be visible to python.

1. Create a new virtual environment

```sh
python3 -m venv .venv
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${PWD}:${PWD}/ImageBind"
```

2. Install all dependencies specified in the `requirements.txt` file

```sh
pip3 install -r requirements.txt
```

3. Install additional dependencies needed for running tests

```sh
pip3 install -r requirements-test.txt
```

4. Download the ImageBind model locally

```sh
python3 download.py
```

5. Run the inference server

```sh
uvicorn app:app --host 0.0.0.0 --port 8000
```

💡 Testing
----------

For sanity checks that to check that all works properly you can run our smoke tests against your server

```sh
python3 smoke_tests.py
```

🐳 Docker support
-----------------

In order to build locally a docker image one can run this command in project's root folder

```sh
LOCAL_REPO="local-bind" ./cicd/build.sh
```

In order to test the built docker image run this command in project's root folder

```sh
LOCAL_REPO="local-bind" ./cicd/test.sh
```

🔗 Useful Resources
--------------------

- [Meta AI ImageBind annoucement article](https://ai.facebook.com/blog/imagebind-six-modalities-binding-ai/)
- [ImageBind github project](https://github.com/facebookresearch/ImageBind)
