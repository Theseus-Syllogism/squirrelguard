# Console: install, configure and run

One tool, three ways to use it. All three share the same code (`squirrel_guard/console/core.py`), so a setting
changed in one shows up in the others.

| | Start it with | Use it when |
|---|---|---|
| Commands | `./sg status`, `./sg check`, `./sg config set ...` | scripting, slow SSH lines, cron |
| Terminal UI | `./sg` | working on the machine or over SSH, no browser needed |
| Web GUI | `./sg gui` | a browser is at hand; reaches a remote machine through an SSH tunnel |
| Install menu | `./sg setup` | first installation, and any time a step needs redoing |

The console needs only Python 3.11 or newer. It uses nothing outside the standard library, because it has to run
before anything else is installed.

## Roles

A machine does one of three jobs. Pick it once (`./sg role detector`, or on the Install page):

| Role | What it runs | Install steps |
|---|---|---|
| `detector` Detector workstation | the detector for one camera stream, as a systemd user service | Python environment, packages, configuration, model export, stream address, deterrent sound, service, start |
| `recorder` Recorder host | the container stack in `deploy/recorder` | Docker check, copy the stack files, database address, `setup.sh` (passwords, tables, build, start), start |
| `gpu` GPU server | Frigate and the listener in `deploy/gpuserver` | Docker check, camera list (by hand), start |

### Detector in a container

The detector can run as a Docker container instead of a Python environment on the machine:

```
./sg runtime docker        # back again: ./sg runtime venv
./sg setup                 # Docker check, configuration, stream address, build image, export model, start
```

The same switch is on the Install page of the web GUI and the `m` key in the terminal UI. Status, start, stop,
restart, logs and checks then work on the container. Files: `Dockerfile`, `compose.yaml`, `.dockerignore` in the
checkout; by hand it is `docker compose up -d --build detector`.

- The image (about 900 MB) holds only the code. `config.toml`, `models/`, `sounds/` are mounted read-only,
  `data/` read-write, and secrets come from `.env` at start: nothing private is built into the image.
- The container runs as your user, with a read-only root filesystem, no capabilities and no privilege escalation.
- `docker compose --profile tools run --rm export` builds a one-off tools image (PyTorch, about 2 GB) and writes
  the model to `models/`. Skip it if you already have the model file.
- The deterrent command runs inside the container, which has no sound device. Keep `actuator.dry_run` on, use a
  command that reaches a network speaker, or mount the host's PulseAudio socket yourself.

Log lines shown by the console have passwords inside addresses masked (ffmpeg prints the stream address when it
cannot connect).

Every step shows the exact commands before it runs, reports whether it is already done, and can be run again.
`./sg setup --dry-run` prints what each step would do and changes nothing.

## Install menu

```
./sg setup                      # full-screen menu (same as the Install page of the terminal UI)
./sg setup --plain              # one question per step; works on serial consoles
./sg setup --dry-run            # show every step, change nothing
./sg setup --yes --role detector   # unattended; steps that need a secret are skipped
```

Steps that need a secret (the camera stream address, the database address) ask for it with hidden input and store
it in a private file (mode 600). In unattended runs, set them first with `./sg env set SG_RTSP_URL`.

## Commands

```
./sg status [--json]            summary: role, install progress, checks, service, last event
./sg check [--role R] [--json]  preflight checks; exit code 1 if any fail
./sg config list [SECTION]      all settings, with defaults marked
./sg config get camera.fps
./sg config set camera.fps 5    checked, then saved; the previous file is kept as config.toml.bak
./sg env list                   which secrets are set (values are never shown)
./sg env set SG_RTSP_URL        asks for the value, hidden
./sg env unset SG_MQTT_PASS
./sg endpoints list [--role R]  outside servers: address, model name, API key set or not
./sg endpoints test [ID]        test one or all; exit code 1 if any fail
./sg endpoints set verifier http://127.0.0.1:8080
./sg endpoints key verifier     asks for the API key, hidden (`key verifier -` removes it)
./sg endpoints model verifier qwen-vl
./sg service status|start|stop|restart|enable|disable|logs [-n 80]
./sg events [-n 20] [--json]
./sg role [detector|recorder|gpu]
./sg -c /path/to/config.toml ...   work on another configuration file
```

## Terminal UI

`./sg` opens a menu on the left and the chosen page on the right. The line under the list explains the selected
row; the bottom line lists the keys for the page.

| Key | Does |
|---|---|
| Up/Down or `j`/`k` | move |
| Enter or Right | open the page, run the step, change the setting |
| Left or Esc | back to the menu |
| `1`–`8` | jump to a page from the menu |
| `r` | refresh |
| `?` | help |
| `q` | quit |

Install steps and secret entry leave the full-screen view for the plain terminal, so the output can be scrolled
back and a secret is never drawn on screen. On the Install page, `d` switches dry run on and off, `a` runs all
remaining steps, `o` changes the role. On the Endpoints page, `t` tests the selected server, `a` tests all of
them, `x` removes an API key.

## Web GUI

```
./sg gui                        # prints an address and opens it in the browser
./sg gui --no-open --port 8760  # on a machine without a browser
ssh -L 8760:127.0.0.1:8760 user@machine   # then open the printed address on your own computer
```

Pages: Overview, Install, Configuration, Endpoints, Secrets, Checks, Service, Events. It uses the same design system as the
review interface (`deploy/recorder/persons-ui/static/ui.css`), including the light/dark switch.

Because it can change settings, store secrets and run install steps, it is restricted:

- It listens on `127.0.0.1` only. There is no option to expose it; use an SSH tunnel.
- The printed address carries a random key, valid until the console is stopped. The key becomes an HttpOnly,
  SameSite=Strict cookie; requests without it are refused.
- Requests whose Host header is not the loopback address are refused (DNS rebinding). Changes also need the key in
  a request header and are refused when they come from another site's page.
- It runs only the fixed list of install steps. There is no "run a command" endpoint.
- Secret values are accepted and stored but never sent back. The stream check reports "reachable" without naming
  the host.

## Endpoints

The servers an installation talks to, per role, each with its address, a **Test** button and, where it applies, an
API key and a model name:

| Role | Endpoint | Stored in |
|---|---|---|
| detector | second-opinion model (`verifier`) | `config.toml` `verifier.url`, `verifier.model`; key `SG_VERIFIER_API_KEY` in `.env` |
| detector | message broker (`mqtt`) | `.env` `SG_MQTT_HOST`, `SG_MQTT_PORT` |
| gpu | second-opinion model, description model (`profiler_llm`), Frigate API, review link, broker | `deploy/gpuserver/squirrel-guard/config.toml`; keys and broker in `deploy/gpuserver/sg.env` |
| recorder | timestamp service (`tsa`) | `TSA_URL` in the stack's `.env` (absent: DigiCert; empty: no outside timestamps) |

Model servers can be llama.cpp (`llama-server`) or anything else with an OpenAI-compatible `/v1/chat/completions`
that accepts images. Give the server's base address, without `/v1`. The test reads `/health` (a server still
loading its model is a warning), `/v1/models` (the model list, which also fills the model-name choices) and, on
llama.cpp, `/props`: a model loaded without its `--mmproj` file cannot see images and fails the test. A server
started with `--api-key` needs the key here; it is stored in the private env file, sent as `Authorization: Bearer`,
and never shown. The model name matters only for model routers and non-llama.cpp servers. The description model
falls back to the second-opinion model's address, key and model when its own are not set.

Container names such as `http://frigate:5000` resolve only inside the stack's network, so they are reported as not
testable from the console; `host.docker.internal` is tested as this machine. The timestamp test asks for one real
timestamp of a fixed test hash. Changes take effect when the service or the containers restart.

## Configuration editing

Settings are edited line by line, so comments and layout in `config.toml` survive. A change is refused, and the
file left untouched, when the value has the wrong type, is out of range, or would leave the file unparseable.
Settings that span several lines (and arrays of tables) are shown but must be changed in a text editor.
Limits live in `RULES` in `core.py`; a setting not listed there is checked for type only.

## Tests

```
.venv/bin/python -m pytest tests/test_console.py tests/test_endpoints.py
SG_ROOT=/path/to/a/scratch/checkout ./sg ...     # point the console at another folder
```

What is tested: configuration round trips, refusal of bad values, secrets staying private, dry runs changing
nothing, the stack copy keeping existing secrets, and the web GUI's access rules. The recorder's `setup.sh` step
and the GPU server's start step run existing scripts and are covered by dry run only.
