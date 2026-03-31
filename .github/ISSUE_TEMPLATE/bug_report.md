---
name: Bug report
about: Report a bug and help us improve stochtree
title: "[BUG]"
labels: bug
assignees: ''

---

# Description
A clear and concise description of what the bug is. Please include details on whether you're using R or Python and which model you're using (i.e. BART, BCF, custom model).

# Reproducing
If possible, include the precise steps to reproduce the behavior. If not, just describe in as much detail how this error showed up for you. Something like
1. Generate data via 
```
set.seed(1234)
```
2. Run BART via 
```
bart_model = bart(X_train = X, y_train = y, ...)
```
3. See error

# Expected behavior
A clear and concise description of what you expected to happen.

# Screenshots
If applicable, add screenshots to help explain your problem.

# System (please complete the following information):
 - OS: [e.g. iOS]
 - Browser [e.g. chrome, safari]
 - Version [e.g. 22]

# Additional context
Add any other context about the problem here.
