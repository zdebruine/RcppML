## Package and options
library("MASS")
options(prompt = "R> ", continue = "+  ", width = 70,
  useFancyQuotes = FALSE)


## Data
data("quine", package = "MASS")


## Visualization
par(mar = c(4, 4, 1, 1))
plot(table(quine$Days), xlab = "Days", ylab = "Frequency", axes = FALSE)
axis(2)
axis(1, at = 0:16 * 5, labels = FALSE)
axis(1, at = 0:8 * 10)
box()


## Poisson model
m_pois <- glm(Days ~ (Eth + Sex + Age + Lrn)^2, data = quine,
  family = poisson)
summary(m_pois)


## Negative binomial model
library("MASS")
m_nbin <- glm.nb(Days ~ (Eth + Sex + Age + Lrn)^2, data = quine)
summary(m_nbin)


## Comparison
BIC(m_pois, m_nbin)
