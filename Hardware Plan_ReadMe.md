# PPE Detection Robot Hardware Plan

## System Overview

The robot will use two main processors:

- **Raspberry Pi 5** for camera input, YOLO PPE detection, logging, and high-level decisions.
- **STM32** for motor control, sensors, wheel encoders, and robot navigation.

```text
Camera
  |
  v
Raspberry Pi 5
  |
  | PPE Detection / YOLO
  |
  v
UART Communication
  |
  v
STM32
  |
  | Motor Control
  | Ultrasonic Sensors
  | Wheel Encoders
  |
  v
Motor Driver
  |
  v
4WD Robot Chassis
```

---

## Recommended Hardware

| Part | Purpose | Estimated Cost |
|---|---|---:|
| Raspberry Pi 5 4GB | YOLO, OpenCV, camera processing, logging | ~$70–100 |
| Existing STM32 | Motors, sensors, navigation | $0 |
| 4WD encoder chassis | Robot platform and motors | ~$25–60 |
| 1080p USB camera | PPE detection video feed | ~$20–30 |
| TB6612FNG motor driver | Controls DC motors from STM32 | ~$10–15 |
| Ultrasonic sensors | Obstacle detection | ~$10–20 |
| Battery + charger | Powers robot | ~$25–40 |
| Voltage regulators / wiring | Power distribution | ~$15–25 |
| Mounts, wires, switches, misc. | Assembly | ~$20–30 |

### Optional

| Part | Purpose | Estimated Cost |
|---|---|---:|
| Raspberry Pi AI HAT+ 13 TOPS | Faster YOLO inference | ~$70 |

The AI HAT+ should only be added if the Raspberry Pi 5 cannot meet the required PPE detection frame rate.

---

## Estimated Budget

### Without AI Accelerator

```text
Raspberry Pi 5        ~$80
Camera                 $25
4WD Chassis            $30
Motor Driver           $10
Sensors                $15
Battery / Power        $35
Miscellaneous          $20
--------------------------------
Estimated Total       ~$215
```

### With AI HAT+

```text
Raspberry Pi 5        ~$80
AI HAT+                $70
Camera                 $25
4WD Chassis            $30
Motor Driver           $10
Sensors                $15
Battery / Power        $35
Miscellaneous          $20
--------------------------------
Estimated Total       ~$285
```

### Target Budget

- **Basic working robot:** ~$200–220
- **Final robot with AI accelerator:** ~$280–300
- **Recommended project budget:** ~$250–300

---

## Raspberry Pi Responsibilities

The Raspberry Pi will handle:

- Camera input
- OpenCV
- YOLO PPE detection
- Person detection
- Detection logging
- Web dashboard
- High-level robot decisions
- Communication with the STM32

Possible PPE classes include:

- Person
- Hard hat
- Safety vest
- Safety glasses

---

## STM32 Responsibilities

The STM32 will handle:

- DC motor control
- PWM motor speed
- Motor direction
- Ultrasonic sensors
- Wheel encoders
- Obstacle avoidance
- Emergency stop behavior
- Low-level navigation
- Commands received from the Raspberry Pi

Example movement functions:

```c
move_forward();
move_backward();
turn_left();
turn_right();
stop();
```

The Raspberry Pi can send simple UART commands such as:

```text
F = Forward
B = Backward
L = Left
R = Right
S = Stop
```

---

## Why Use Wheel Encoders?

Two motors given the same PWM value may not spin at exactly the same speed.

Without encoder feedback:

```text
Left Motor  = 50% PWM
Right Motor = 50% PWM

Left Speed != Right Speed

Result: Robot drifts
```

With encoders:

```text
Wheel Encoder
     |
     v
STM32 Measures Wheel Speed
     |
     v
Adjust PWM
     |
     v
Robot Drives Straighter
```

---

## Development Plan

### Phase 1 — PPE Detection

Set up the Raspberry Pi with:

- Camera
- Python
- OpenCV
- YOLO

Goal:

```text
Camera
  |
  v
Raspberry Pi
  |
  v
Detect Person
  |
  v
Detect PPE
```

---

### Phase 2 — STM32 Motor Control

Connect the STM32 to the motor driver.

Implement and test:

```c
move_forward();
move_backward();
turn_left();
turn_right();
stop();
```

---

### Phase 3 — Robot Navigation

Add:

- Ultrasonic sensors
- Wheel encoders

The STM32 should be able to handle basic obstacle avoidance without relying on the Raspberry Pi.

```text
Ultrasonic Sensor
       |
       v
     STM32
       |
       v
Obstacle?
   /      \
 Yes       No
  |         |
Stop/Turn Continue
```

---

### Phase 4 — Raspberry Pi to STM32 Communication

Connect the Raspberry Pi and STM32 through UART.

Example:

```text
Raspberry Pi
     |
     | Sends "F"
     v
   STM32
     |
     v
Move Forward
```

---

### Phase 5 — Full Integration

```text
                Camera
                  |
                  v
          Raspberry Pi 5
          /      |      \
       YOLO   Logging   Dashboard
          |
          v
     Robot Decision
          |
         UART
          |
          v
        STM32
       /  |   \
 Motors Sensors Encoders
       |
       v
 Motor Driver
       |
       v
    4WD Robot
```

---

## Reference Videos

Useful YouTube searches for STM32 robot development:

- [STM32 Obstacle Avoiding Robot](https://www.youtube.com/results?search_query=STM32+obstacle+avoiding+robot+car+tutorial)
- [STM32 4WD Robot Car](https://www.youtube.com/results?search_query=STM32+4WD+robot+car+tutorial)
- [STM32 Nucleo Robot Car](https://www.youtube.com/results?search_query=STM32+Nucleo+robot+car+tutorial)
- [STM32 + L298N Robot Car](https://www.youtube.com/results?search_query=STM32+L298N+robot+car+tutorial)
- [STM32 Ultrasonic Autonomous Robot](https://www.youtube.com/results?search_query=STM32+ultrasonic+autonomous+robot+car)

---

## Recommended Starting Setup

Start with:

1. Raspberry Pi 5 4GB
2. Existing STM32
3. 4WD chassis with wheel encoders
4. Cheap 1080p USB camera
5. TB6612FNG motor driver
6. Ultrasonic sensors
7. Battery and voltage regulation hardware

Do not buy the AI HAT+ immediately.

First test the PPE model on the Raspberry Pi 5. If performance is too slow, add the AI accelerator later.