### Installation
1. Install conda (if not installed) https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

2. Create conda environment with python=3.10.12 and activate it

    ```
    conda create -n <env_name> python=3.10.12
    conda activate <env_name>
    ```

3. Install dependencies
    ```
    pip install hebi-py pytorch-kinematics pytorch-minimize pybullet trimesh "pyglet<2"
    ```

4. To use ZED 2i stereo camera (for positional tracking)
    - download ZED SDK with corresponding CUDA version https://www.stereolabs.com/en-sg/developers/release
    - install ZED Python API https://www.stereolabs.com/docs/app-development/python/install
        ```
        python -m pip install opencv-python pyopengl requests
        cd /usr/local/zed
        python get_python_api.py
       ```
    - note: if ```ImportError: version `GLIBCXX_3.4.30' not found (required by /usr/local/zed/lib/libsl_zed.so)```, do: 
        ```
        conda install -c conda-forge libstdcxx-ng
        ```
