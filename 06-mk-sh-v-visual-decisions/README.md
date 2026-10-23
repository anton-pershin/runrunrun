# MK-SH-V: quick visual-decision probe

## Question

Can MK-SH-V, a small image-plus-text decision model, answer practical questions about arbitrary screenshots and memes, including a basic object/scene inference, text presence, and counting people?

This is a three-example exploratory probe, not a benchmark. The goal was to quickly test the model's useful capabilities and failure modes before deciding whether further evaluation is worthwhile.

## Setup and method

- Model/repository: [Gjusev/mk-sh-v](https://github.com/Gjusev/mk-sh-v), checkpoint `Gjusev/MK-SH-V` from Hugging Face.
- The model was deployed locally on CPU in `/home/tony/mk-sh-v` using Python 3.11 and the repository's Gradio demo. The demo API exposes its calibrated probabilities, choice, and abstention threshold (`tau = 0.55`).
- For each user question, the image, a short state description, the question, and candidate options were passed to the model. Images and exact questions are preserved in `images/` and the case log below.
- The demo permits at most 8 options. The count question had 11 possible integer answers (0–10), so it was probed in two stages with reduced candidate sets. This changes the task and is a limitation of that observation.
- Results below are the actual local model outputs from the conversation. Human visual inspection is noted separately and is not treated as model output.

## Results

### Case 1 — map marker and island

Image: [`images/01-karedzhi-map.jpg`](images/01-karedzhi-map.jpg)

Question: **“Is the red arrow located on an island?”**

Options: `yes: The arrow is on an island`; `no: The arrow is not on an island`.

Model result: **no**, confidence **86.0%**, above `tau = 0.55`.

Visual check: the marker appears over the area labeled “о. Кареджи” (“Karedzhi Island”). The best-supported answer is **yes**. The model gave a confident wrong answer.

### Case 2 — text/number presence

Image: [`images/02-order-screen.jpg`](images/02-order-screen.jpg)

Question: **“Does the image contain the number 2298?”**

Options: `yes: The image contains the number 2298`; `no: The image does not contain the number 2298`.

Model result: **yes**, confidence **73.2%**, above `tau = 0.55`.

Visual check: the large order number `2298` is visible inside the red circle. The model answered correctly. This is a straightforward text-presence case; it does not establish general OCR reliability.

### Case 3 — counting people in a meme

Image: [`images/03-tom-delonge-meme.jpg`](images/03-tom-delonge-meme.jpg)

Question: **“How many people does the image contain? The answer is a number from 0 to 10.”**

A direct 0–10 candidate list was rejected by the demo because it allows at most eight options. We then used two calls:

1. Options `0-3: Zero to three people`, `4-6: Four to six people`, `7-10: Seven to ten people`. Model chose **0–3**, confidence **93.2%**.
2. Options `3: Three people`, `4: Four people`, `5: Five people`, `6: Six people`. Model chose **3**, confidence **90.6%**.

Visual check: the collage contains Tom DeLonge and portraits of Karl Marx, Friedrich Engels, and Vladimir Lenin: **4 people**. The staged model result was wrong, and both stages were confident. Because the second call's candidate set was selected independently rather than strictly conditioned on stage one's chosen range, this should be read as a practical probing workaround, not a principled multi-stage classifier. The demo's eight-option cap also prevented a single direct evaluation of all 11 requested answers.

## Findings

- The model handled the clearest text-presence question, returning the correct number-presence answer at 73.2%.
- It failed the island-location question with a confident wrong answer (86.0%).
- It failed the people-counting probe, also with high confidence at both stages (93.2% for the wrong range and 90.6% for the wrong exact count).
- The configured refusal threshold did not protect against these two errors: each selected answer was above `tau = 0.55`. The repository's published calibration claim is distribution-limited; these few arbitrary real images do not validate or refute that evaluation, but they show that the gate should not be treated as a general-purpose correctness guarantee.
- This is a tiny, hand-picked, non-blinded sample of three cases. It supports a qualitative conclusion only: useful on a simple visible-text check, unreliable on the tested spatial/geographic and counting questions. Do not infer benchmark-level accuracy, general OCR performance, or broad model capability from these examples.

## Reproduction

1. Clone [the upstream repository](https://github.com/Gjusev/mk-sh-v) and follow its README/demo setup. This run used a local CPU installation and the released `Gjusev/MK-SH-V` checkpoint.
2. Start the demo with `MKSHV_MODEL_DIR=/path/to/model-and-policy .venv/bin/python demo/app.py` and submit each image, state, question, and option list shown above.
3. For the count case, use the two option sets recorded above. Keep in mind that this is a staged workaround, not an 11-way single-call result.

The model repository and checkpoint are external dependencies and are not copied into this study. The three user-provided test images are included here so the examples can be inspected alongside their questions.
