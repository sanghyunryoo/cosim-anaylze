<div align="center">
  <h1>cosim</h1>
  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge">
    <img src="https://img.shields.io/badge/MuJoCo-3.2.7-FF6F00?style=for-the-badge&logo=openai&logoColor=white" alt="MuJoCo Badge">
    <img src="https://img.shields.io/badge/PyQt5-5.15.11-41CD52?style=for-the-badge&logo=qt&logoColor=white" alt="PyQt5 Badge">
  </p>
</div>

## Overview

<p align="center">
  <img src="docs/img/main_ui_img.png" alt="Main UI" width="65%">
  <img src="docs/img/report_sample_img.png" alt="Report Sample" width="30%">
</p>

`cosim` is a MuJoCo-based sim-to-sim testing tool with a PyQt GUI.  
It is mainly used to load an ONNX policy, choose an environment, tune simulation and hardware settings, run rollouts, and inspect results with reports and motor time-series plots.

## Main Features

- Run multiple MuJoCo environments from one GUI
- Load ONNX policies and test them with keyboard command input
- Tune actuator, hardware, observation, and initial pose settings per environment
- Apply randomization such as friction, mass noise, load, and action delay
- Use built-in terrain presets such as `flat`, `rocky_*`, `slope_*`, and `stairs_*`
- Inspect motor torque and velocity in a detached real-time monitor window
- Show an end-of-test summary plot for selected joints
- Generate a PDF report after rollout

## Installation

### 1. Create an environment

Python `3.10` is recommended.

```bash
conda create -n cosim python=3.10
conda activate cosim
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the GUI

```bash
python launch.py
```

## Basic Usage

### 1. Select an environment

Choose an environment from the `Environment Settings` panel.

Environment definitions live in [config/env_table.yaml](/home/sanghyunryoo/Documents/4w4l/cosim_act_net/cosim_act_net/config/env_table.yaml).

### 2. Load a policy

In `Policy Settings`:

- choose the policy type
- set hidden-state dimensions if needed
- select the ONNX file with `Browse`
- if you use `Encoder+MLP`, also select the encoder ONNX file

### 3. Adjust settings if needed

You can open these dialogs from the GUI:

- `Actuator Settings`
- `Action Scale Settings`
- `Hardware Settings`
- `Initial Pose Settings`
- `Observation Settings`

Randomization controls are available in `Random Settings`.

### 4. Choose terrain and command mode

In `Environment Settings`:

- set the terrain
- set max duration
- enable or disable position command mode if needed

### 5. Start a rollout

Click `Start Test`.

During rollout:

- use the command buttons or keyboard shortcuts to send commands
- watch logs in `Terminal Log`
- stop the test with `Stop Test`

## Motor Monitor

The motor monitor is configured from the `Monitor` row in `Environment Settings`.

- `Joints`: choose which joints to track
- `Window`: open the detached real-time motor monitor during test
- `Show End`: open a summary plot window automatically when the test finishes, errors, or ends after viewer close

### Real-time monitor behavior

- The monitor window does not open until a test is actually running
- Closing the monitor window with `X` also turns off the `Window` checkbox
- Torque and velocity plots use dynamic scaling so recent motion is easier to see

### End-of-test summary

When `Show End` is enabled, a Qt window with matplotlib plots is shown at the end of the run.

## Report Output

After a successful run, the GUI can open the generated `report.pdf`.

The report is typically saved next to the selected policy file.

## Configuration Files

### Environment table

[config/env_table.yaml](/home/sanghyunryoo/Documents/4w4l/cosim_act_net/cosim_act_net/config/env_table.yaml)

Contains:

- available environments
- observation order and scales
- command scales
- action scales
- hardware gains and torque or velocity limits

### Randomization table

[config/random_table.yaml](/home/sanghyunryoo/Documents/4w4l/cosim_act_net/cosim_act_net/config/random_table.yaml)

Contains:

- simulation precision presets
- sensor noise presets

## Adding a New Environment

To add a new environment, update all of the following:

1. Add the environment class under `envs/<env_name>/`
2. Register it in [envs/build.py](/home/sanghyunryoo/Documents/4w4l/cosim_act_net/cosim_act_net/envs/build.py)
3. Add its config block to [config/env_table.yaml](/home/sanghyunryoo/Documents/4w4l/cosim_act_net/cosim_act_net/config/env_table.yaml)
4. Add initial pose metadata to [envs/initial_pose.py](/home/sanghyunryoo/Documents/4w4l/cosim_act_net/cosim_act_net/envs/initial_pose.py) if needed
5. Make sure the XML, meshes, terrains, and sensor names match the environment code

## Project Structure

```text
cosim/
├── core/                  # Policy interface, tester, reporting
├── envs/                  # MuJoCo environments
├── config/                # YAML configs
├── docs/                  # Images and documentation assets
├── ui/                    # PyQt GUI
├── launch.py              # GUI entry point
└── README.md
```

## Notes

- MuJoCo rendering and PyQt windows require a desktop environment
- If a new environment behaves strangely, verify actuator order, observation order, sensor naming, and initial pose first
- For sim-to-sim checks, small mismatches in joint ordering or action ordering can break transfer badly
