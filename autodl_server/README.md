# AutoDL LLaVA Server

This service accepts a camera image at `POST /locate` and returns a drone
bounding box and center in original-image pixel coordinates.

## 1. Install On AutoDL

Upload this directory to the AutoDL instance, then install the server and model
dependencies in the Python environment that can load the LLaVA model:

```bash
cd /root/autodl-tmp/autodl_server
python3 -m pip install -r requirements.txt
```

## 2. Start In Mock Mode

Mock mode verifies image upload and JSON responses without loading LLaVA:

```bash
cd /root/autodl-tmp/autodl_server
LLAVA_MOCK=true uvicorn llava_server:app --host 127.0.0.1 --port 8000
```

Keep that terminal running. On the machine running Gazebo, open an SSH tunnel:

```bash
ssh -N -L 8000:127.0.0.1:8000 USER@AUTODL_HOST -p AUTODL_SSH_PORT
```

Verify the tunnel locally:

```bash
curl http://127.0.0.1:8000/health

ffmpeg -y -i drone_flight_720p.mp4 -frames:v 1 /tmp/drone_test.jpg
curl -X POST \
  -F request_id=1 \
  -F image=@/tmp/drone_test.jpg \
  http://127.0.0.1:8000/locate
```

The mock response center is the exact center of the uploaded image.

## 3. Start With LLaVA

Stop the mock service and provide the model directory:

```bash
cd /root/autodl-tmp/autodl_server
LLAVA_MODEL_PATH=/root/autodl-tmp/MODEL_DIRECTORY \
  uvicorn llava_server:app --host 127.0.0.1 --port 8000
```

Load a PEFT LoRA adapter directly without exporting or merging the base model:

```bash
LLAVA_MODEL_PATH=/path/to/llava-1.5-7b-hf \
LLAVA_ADAPTER_PATH=/path/to/drone-lora \
  uvicorn llava_server:app --host 127.0.0.1 --port 8000
```

For models whose Hugging Face implementation requires custom Python code:

```bash
LLAVA_MODEL_PATH=/root/autodl-tmp/MODEL_DIRECTORY \
LLAVA_TRUST_REMOTE_CODE=true \
  uvicorn llava_server:app --host 127.0.0.1 --port 8000
```

The provided backend uses `AutoProcessor` and `AutoModelForVision2Seq`. Some
original LLaVA repositories use a different loader and conversation template;
in that case only `LlavaBackend` needs to be adapted. The HTTP contract and the
local ROS bridge remain unchanged.

## Current AutoDL Instance

The verified deployment uses:

- SSH: `root@connect.bjb2.seetacloud.com`, port `22767`
- model: `/root/llava`
- LoRA: `/root/drone-lora-2026-02-02`
- Python: `/root/drone-llava-env/bin/python`
- service: `/root/drone-llava-service`

Start or restore the remote service from the local machine:

```bash
ssh -i /home/rzfly/.ssh/id_ed25519_autodl_llava \
  -p 22767 root@connect.bjb2.seetacloud.com

cd /root/drone-llava-service
nohup env \
  LLAVA_MODEL_PATH=/root/llava \
  LLAVA_ADAPTER_PATH=/root/drone-lora-2026-02-02 \
  PYTHONUNBUFFERED=1 \
  /root/drone-llava-env/bin/python \
  -m uvicorn llava_server:app --host 127.0.0.1 --port 8000 \
  > server.log 2>&1 < /dev/null &
echo $! > server.pid
```

Start the local tunnel in another terminal:

```bash
ssh -N -L 8000:127.0.0.1:8000 \
  -i /home/rzfly/.ssh/id_ed25519_autodl_llava \
  -p 22767 root@connect.bjb2.seetacloud.com
```

The current AutoDL SSH port can change when the instance is recreated. Replace
`22767` with the port shown in the AutoDL console when necessary.

The service requires CUDA by default and exits immediately in AutoDL no-GPU
mode. This avoids accidentally loading the 7B base model into CPU memory. The
2 February LoRA adapter remains separate from the base model and is attached in
memory by PEFT when the service starts.

## API Contract

`GET /health` returns the active mode and model path.

`POST /locate` accepts multipart form fields:

- `image`: JPEG or PNG image
- `request_id`: integer copied into the response
- `prompt`: optional instruction; defaults to the deployed model prompt
- `do_sample`: `true` or `false`, default `true`
- `temperature`: positive number, default `0.95`
- `top_p`: number in `(0, 1]`, default `0.7`
- `max_new_tokens`: integer in `1..1024`, default `128`

These generation fields are applied per request. Changing them does not restart
the service or reload the base model and LoRA.

Successful detection response:

```json
{
  "found": true,
  "bbox": [512.0, 288.0, 768.0, 432.0],
  "center": [640.0, 360.0],
  "request_id": 1,
  "image_width": 1280,
  "image_height": 720,
  "generation": {
    "do_sample": true,
    "temperature": 0.95,
    "top_p": 0.7,
    "max_new_tokens": 128
  },
  "raw_answer": "..."
}
```
