# Tiny Robot LM

A 3.2M-parameter language model, built from scratch, that turns typed instructions into robot movement. It runs fully offline on a Jetson Nano.

**Winner of MarinaHacks 2026 at CSULB.**

[Devpost](https://devpost.com/software/tiny-robot-lm) · [Demo video](https://www.youtube.com/watch?v=Lmy4Fd--4fg)

## Why we built it

This started as a summer project: an autonomous food delivery robot. The first version grabbed a camera frame every five seconds, sent it to a cloud LLM, and waited for a command back. That was slow and expensive, and the robot couldn't do anything without an internet connection.

We wanted the robot to make its own decisions on its own hardware. A large model wasn't going to fit on a Jetson Nano, so we trained a tiny one.

## How it works

The work is split between two models:

1. A pre-trained CNN (YOLO) watches the camera stream and detects objects.
2. Tiny Robot LM reads the user's instruction and translates it into a short action code.

The robot's control code combines the action code with the CNN's detections and drives the motors. For example:

```
"move forward until you see a bottle"  ->  FWD UNTIL bottle
```

Image recognition stays with the CNN, which is already good at it. The language model only has to understand commands, so it can stay small enough to run on the robot without a GPU, with almost no noticeable latency. Since we own the model, we can also change the command vocabulary whenever we need to.

## The model

| | |
|---|---|
| Parameters | 3.2M |
| Layers | 4 |
| Attention heads | 4 |
| Size on disk | ~13 MB |
| Training | 30 epochs, batch size 128 |
| Accuracy | 93.8% |

### Training data

We used a large LLM to brainstorm example instructions, then wrote our own script to split, fill, filter, balance, and add noise to that output. The result was 40,000 training examples. Labels came from our script rather than the LLM, and we checked them by hand.

## Built with

- Python
- Transformers
- YOLO
- NVIDIA Jetson Nano
- HTML, CSS, JavaScript

Motor bridging and vehicle control reuse the Jetson Nano's open-source code.

## Challenges

Our first plan was a small language model. We quickly realized that would need millions of training examples and far more compute and memory than we had in 24 hours. Shrinking to a tiny model with a narrow job made the project possible.

The Jetson Nano we used is an older board, which shaped a lot of our choices. Every command the model outputs has to be one the robot can safely execute.

## What we learned

Data quality mattered more than data quantity. Writing our own labeling script and reviewing the output by hand did more for accuracy than generating extra examples.

## What's next

The goal is a delivery robot that navigates indoors with no network connection and takes instructions like "Deliver this parcel from ECS 416 to DESN 112." The next step is training the CNN to recognize walls, doors, and other obstacles.

## Team

- [Stephanie Noe](https://devpost.com/stephkmn)
- [Gokila Harini Krishna](https://devpost.com/gokilaharinik)

Built in 24 hours at MarinaHacks 2026, California State University, Long Beach.
