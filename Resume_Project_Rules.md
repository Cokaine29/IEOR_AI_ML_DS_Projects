# Rules for Building & Writing Resume Projects

Derived from the pattern across your seniors' IITB placement resumes. Apply these to every
project — the 4 new AI/ML ones, the existing OR ones, and eventually SDE.

---

## 1. Relevance rules

- **Every project must map to the resume it's on.** Don't include a project on a resume
  just because it exists — an NLP fake-review detector doesn't belong on your OR/SC resume
  even if it's impressive. Match project to the skills that resume's recruiters are
  screening for.
- **A project that fits two resumes is a bonus, not the goal.** Don't force a stretch
  connection (e.g., don't claim your fault-detection project is "SDE" just because it has
  a Python script). Cross-listing only works when the core technical method genuinely
  overlaps with what that track's recruiters test for.
- **No filler projects.** If a project doesn't clearly demonstrate a skill the JD or
  interview will test, cut it — a resume with 4 strong projects beats one with 6 where 2
  are weak. Quality over count, even though seniors show 5-6.

## 2. Bullet structure rules

- **3 bullets per project, no more, no less** (unless the project is genuinely bigger —
  MTP/Thesis can flex to 4-5).
- **Bullet 1 = what/objective.** What you built and why (the real problem in one clause).
- **Bullet 2 = how/method.** The technique, architecture, pipeline, or approach — this is
  where most of your bolded technical terms live.
- **Bullet 3 = result.** A number, ideally compared against a baseline.
- **No filler sentences.** Every bullet should be information-dense — if a clause doesn't
  add a technical detail or a result, cut it.

## 3. Quantification rules — the most important section

- **Every bullet ends in a number.** No exceptions. If a bullet doesn't have one, it's not
  finished — go back and measure something.
- **A number alone is weak; a number vs. a baseline is strong.** "88.6% accuracy" is fine.
  "88.6% accuracy, a 4.75% improvement over Random Forest" is what actually reads as
  rigorous — it proves you understood what "good" means for that problem, not just that you
  ran a model.
- **Use the right kind of number for the claim:**
  - Model performance → accuracy, F1, AUC, MAPE, RMSE, R²
  - Efficiency/speed → runtime reduction %, latency, convergence speed
  - Business impact → cost/time/risk reduction %, revenue impact
  - Scale → dataset size, number of features, number of entities handled
- **Never fabricate or round up a number to sound better.** You will be asked to defend
  every single number in an interview — if you can't explain how you got it, don't put it
  on the resume. A recruiter probing "how did you compute that 23% reduction?" and getting
  a vague answer is worse than a smaller, honest number.
- **Report real comparisons, even unflattering ones.** If your fancy method didn't beat
  the baseline, that's still a valid, honest result — say so, and explain why in the
  interview. Seniors' resumes include projects where the "obvious" model won; that's fine.

## 4. Bolding rules

- **Bold tool/library/algorithm names, metrics, and dataset specifics** — not generic verbs
  or filler words. Bold "PyTorch," "Autoencoder," "88.6% accuracy," "CWRU bearing dataset" —
  don't bold "developed" or "used."
- **Bold density should make the bullet skimmable in 3 seconds.** A recruiter scanning
  100 resumes should be able to tell what tools you know just from the bolded words.

## 5. Real-problem framing rules

- **State the real-world constraint, not just the technique.** "Trained an autoencoder for
  anomaly detection" is a technique. "Trained an autoencoder for anomaly detection because
  labeled failure data is scarce in industrial settings" is a problem — always frame the
  *why* before the *what*.
- **Use real/standard datasets, not synthetic ones, wherever possible** — CWRU, M5/Walmart,
  real Amazon/Yelp reviews, real stock data. A dummy/toy dataset undercuts the "real
  problem" story even if the method is sound.
- **Identify what you're contributing, not just what you're running.** Running XGBoost on a
  dataset isn't a contribution by itself — comparing methods under a realistic constraint,
  segmenting results by a meaningful subgroup, or finding which approach degrades more
  gracefully IS a contribution. Every project write-up should be able to answer: "what did
  I learn/show that a straightforward baseline run wouldn't have?"

## 6. Tense & status rules

- **Completed projects → past tense** ("Benchmarked," "Achieved," "Reduced").
- **Ongoing projects (MTP, Seminar) → present-continuous** ("Researching," "Developing,"
  "Integrating") — don't claim a finished result on work still in progress.
- **Don't inflate project status.** If you say "achieved 88.6% accuracy," you should
  actually have that number from a real run before it goes on the resume.

## 7. Narrative coherence rules

- **Each resume should read as one coherent specialization, not a scattered list.** E.g.,
  your AI/ML resume should read as "Industrial AI / Smart Manufacturing + broad ML
  fundamentals," not "did GenAI, then CV, then finance, then NLP with no throughline."
  Where possible, tie projects back to your MTP's Industry 4.0 focus in the framing.
- **Don't have two projects that prove the exact same skill.** If two projects both just
  show "I can fine-tune a transformer," cut one — diversify what each project proves
  (unsupervised learning, time-series, optimization, NLP, etc. should each show up once).

## 8. Interview-readiness rules

- **You must be able to defend every number out loud, unscripted.** For each metric on your
  resume, be ready to answer: How was it computed? What's the baseline? What would happen
  if the test set were different? What are the failure modes?
- **Know why you chose your method over the alternatives.** "Why Autoencoder and not
  Isolation Forest?" should have a real answer (e.g., reconstruction error handles
  non-linear degradation patterns better) — not "it was next on my list."
- **Know the limitations of your own project.** Interviewers respect "this approach breaks
  down when X" far more than pretending a project has no weaknesses. Prepare 1 honest
  limitation per project.
- **Be able to explain the real-world impact in one sentence, no jargon.** If a hiring
  manager (not a technical interviewer) asks "what does this project actually do," you
  should have a plain-English answer ready — this is also good practice for HR/culture-fit
  rounds.
- **Rehearse the "why" before the "what."** Interview panels often start with "walk me
  through this project" — lead with the real problem, not the tech stack, or it reads as
  technique-first instead of problem-first thinking.

## 9. Common mistakes to avoid

- Bullets with no number.
- Numbers with no baseline/comparison.
- Vague verbs ("worked on," "helped with," "explored") instead of strong action verbs
  ("architected," "benchmarked," "optimized," "deployed").
- Claiming a still-in-progress project as done.
- Using a dataset/problem so generic (Iris, Titanic, MNIST) that it reads as a tutorial,
  not a project.
- Overloading one resume with projects that all prove the same skill.
- A number you can't explain if asked to defend it in an interview.
