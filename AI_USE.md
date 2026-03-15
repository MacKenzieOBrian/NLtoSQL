# AI Use

I used AI at the start for project scaffolding so I could work out how I wanted
to structure the repo. It helped me think about splitting the project into a
main package, notebooks, scripts, and results folders.

I also used AI for some basic scripts that helped with menial tasks. That was
mostly helper and analysis code such as `scripts/build_final_analysis.py`,
`nl2sql/evaluation/final_pack.py`, `nl2sql/evaluation/simple_stats.py`,
`nl2sql/infra/*`, and some of the notebook support code.

I used it to help create scripts that analysed outputs so I could work out the
next steps. I also used it when the documentation did not quite match my own
setup and I needed help for a specific error or situation. It helped a bit with
data formatting, comment formatting, and string formatting too.

The main parts were mostly based on the literature and were typed, tested, and
run by me in Colab notebooks with print statements for debugging. That includes
the main prompt flow, the guarded SQL path, the ReAct-style loop, and the main
evaluation logic.

Tab autocomplete in the IDE also helped me stay inside the function I was
working on. If I did not understand a suggestion, or if it felt too abstract, I
did not keep it. I only kept suggestions I could follow and check myself.
