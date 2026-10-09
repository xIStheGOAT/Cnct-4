# Connect 4 Photo Solver

A small Streamlit app that accepts a photo of a Connect Four board, tries to detect the
7×6 position with OpenCV, and uses an alpha-beta/iterative-deepening search to recommend
the strongest move it can find within a chosen time limit.

## Install

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the local address Streamlit prints and upload a board photo.

## Notes

The included detector is intentionally lightweight. It works best when the board is
roughly straight-on and the pieces are clearly visible. Vision errors are possible,
so verify the detected 7×6 grid before using the recommendation.

The solver is a strong search engine but is time-limited. It is not a formal proof of
a win for every position. For an exact perfect-play engine, replace the solver with a
fully solved Connect Four bitboard engine.
