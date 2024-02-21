install.packages("dplyr")
install.packages("glm2")
library(glm2)
library(dplyr)

# Wczytanie przetworzonych danych
cleartrain <- read.csv('cleartrain.csv')
cleartest <- read.csv('cleartest.csv')

y_train <- as.matrix(cleartrain$Survived)
x_train <- as.matrix(cleartrain[, 2:ncol(cleartrain)])
y_test <- as.matrix(cleartest$Survived)
x_test <- as.matrix(cleartest[, 2:ncol(cleartest)])

pdf_logit <- function(z) {
  pdf_value <- exp(z) / (1 + exp(z))^2
  colnames(pdf_value) <- "f(Z = z)"
  return(pdf_value)
}

pdf_probit <- function(z) {
  pdf_value <- dnorm(z, mean=0, sd=1, log=FALSE)
  colnames(pdf_value) <- "f(Z = z)"
  return(pdf_value)
}

# Model logitowy - część 1: Oszacowanie modelu logitowego bez restrykcji
# ----------------------------------------------------------------------------------------------------------------------
logit_model <- glm(y_train ~ x_train, family=binomial(link="logit"))
print("Model logitowy bez restrykcji")
print("--------------------------------")

# Obliczenie potrzebnych statystyk dla modelu logitowego
logit_beta              <- as.matrix(logit_model$coefficients)
logit_covariance_matrix <- vcov(logit_model)
logit_average_std_error <- as.matrix(diag(logit_covariance_matrix)) ^ 0.5
logit_beta_t_ratio      <- abs(logit_beta / logit_average_std_error)
logit_beta_p_value      <- 2 * pnorm(logit_beta_t_ratio, mean=0, sd=1, lower.tail=FALSE, log.p=FALSE)
logit_log_likelihood    <- (-logit_model$aic + 2 * logit_model$rank) / 2

summary <- round(cbind(logit_beta, logit_average_std_error, logit_beta_t_ratio, logit_beta_p_value), 3)
colnames(summary) <- c("Beta hat", "Średni błąd standardowy", "Statystyka t", "p-value")
rownames(summary) <- c("const", colnames(x_train))
summary

# Obliczanie trafności modelu dla zbioru uczącego --------------------------------------------------------------

# Obliczenie wartości przewidywanych (y hat), oznaczających prawdopodobieństwo, że dana osoba przeżyje (y = 1)
x_with_constant <- cbind(1, x_train)
logit_x_t_beta <- x_with_constant %*% logit_beta
logit_survival_probability <- exp(logit_x_t_beta)/(1 + exp(logit_x_t_beta))

# Średnia wartość y, na podstawie której stwierdzamy dalej, czy dana osoba przeżyje
p <- colMeans(y_train)
logit_y_hat <- as.matrix(as.numeric(logit_survival_probability > p))

# Obliczenie tabeli trafności
logit_y_comparison <- cbind(y_train, logit_y_hat)
logit_confusion_matrix <- xtabs(formula = ~ logit_y_hat + y_train, data=logit_y_comparison)

print("[Model logitowy bez restrykcji] Tabela trafności dla zbioru uczącego:")
print(logit_confusion_matrix)

# Obliczenie współczynnika R^2
logit_r_squared <- (logit_confusion_matrix[1, 1] + logit_confusion_matrix[2, 2]) / sum(logit_confusion_matrix)
print("[Model logitowy bez restrykcji] Współczynnik R^2 dla zbioru uczącego:")
print(logit_r_squared)

# Obliczenie współczynnika determinacji Efrona
logit_efron_r_squared <- 1 - (sum((y_train - logit_survival_probability) ^ 2) / sum((y_train - mean(y_train)) ^ 2))
print("[Model logitowy bez restrykcji] Współczynnik determinacji Efrona dla zbioru uczącego:")
print(logit_efron_r_squared)

# Obliczenie współczynnika determinacji Zavoina i McKelveya
T <- dim(x_train)[1]
T <- T * 3.289868
avg_survival_probability <- mean(logit_survival_probability)
nominator <- sum((logit_survival_probability - avg_survival_probability) ^ 2)
denominator <- T + nominator
logit_zavoin_mckelvey_r_squared <- nominator / denominator
print("[Model logitowy bez restrykcji] Współczynnik determinacji Zavoina i McKelveya dla zbioru uczącego:")
print(logit_zavoin_mckelvey_r_squared)

# Efekty krańcowe
logit_pdf_values <- pdf_logit(logit_x_t_beta)
logit_partial_effects <- matrix(0, dim(x_train)[1], ncol(x_with_constant))

for (i in 2:ncol(x_with_constant)) {
  logit_partial_effects[, i] <- logit_pdf_values[, 1] * logit_beta[i]
}

logit_mean_partial_effects <- as.matrix(colMeans(logit_partial_effects))
colnames(logit_mean_partial_effects) <- "[Model logitowy bez restrykcji] Średnie efekty krańcowe dla zbioru uczącego"
rownames(logit_mean_partial_effects) <- colnames(x_with_constant)
logit_mean_partial_effects

# Obliczenie trafności modelu dla zbioru testowego --------------------------------------------------------------

# Obliczenie wartości przewidywanych (y hat), oznaczających prawdopodobieństwo, że dana osoba przeżyje (y = 1)
x_with_constant <- cbind(1, x_test)
logit_x_t_beta <- x_with_constant %*% logit_beta
logit_survival_probability <- exp(logit_x_t_beta)/(1 + exp(logit_x_t_beta))

# Średnia wartość y, na podstawie której stwierdzamy dalej, czy dana osoba przeżyje
p <- colMeans(y_test)
logit_y_hat <- as.matrix(as.numeric(logit_survival_probability > p))

# Obliczenie tabeli trafności
logit_y_comparison <- cbind(y_test, logit_y_hat)
logit_confusion_matrix <- xtabs(formula = ~ logit_y_hat + y_test, data=logit_y_comparison)

print("[Model logitowy bez restrykcji] Tabela trafności dla zbioru testowego:")
print(logit_confusion_matrix)

# Obliczenie współczynnika R^2
logit_r_squared <- (logit_confusion_matrix[1, 1] + logit_confusion_matrix[2, 2]) / sum(logit_confusion_matrix)
print("[Model logitowy bez restrykcji] Współczynnik R^2 dla zbioru testowego:")
print(logit_r_squared)

# Obliczenie współczynnika determinacji Efrona
logit_efron_r_squared <- 1 - (sum((y_test - logit_survival_probability) ^ 2) / sum((y_test - mean(y_test)) ^ 2))
print("[Model logitowy bez restrykcji] Współczynnik determinacji Efrona dla zbioru testowego:")
print(logit_efron_r_squared)

# Efekty krańcowe
logit_pdf_values <- pdf_logit(logit_x_t_beta)
logit_partial_effects <- matrix(0, dim(x_test)[1], ncol(x_with_constant))

for (i in 2:ncol(x_with_constant)) {
  logit_partial_effects[, i] <- logit_pdf_values[, 1] * logit_beta[i]
}

logit_mean_partial_effects <- as.matrix(colMeans(logit_partial_effects))
colnames(logit_mean_partial_effects) <- "[Model logitowy bez restrykcji] Średnie efekty krańcowe dla zbioru testowego"
rownames(logit_mean_partial_effects) <- colnames(x_with_constant)
logit_mean_partial_effects

# Model logitowy - część 2: Oszacowanie modelu logitowego z restrykcjami na kolumnach 5:8
# ----------------------------------------------------------------------------------------------------------------------
logit_r_model <- glm(y_train ~ x_train[, c(1:3, 8:9)], family=binomial(link="logit"))
print("Model logitowy z restrykcjami na kolumnach 5:8")
print("-----------------------------------------------")

# Obliczenie potrzebnych statystyk dla modelu logitowego
logit_r_beta              <- as.matrix(logit_r_model$coefficients)
logit_r_covariance_matrix <- vcov(logit_r_model)
logit_r_average_std_error <- as.matrix(diag(logit_r_covariance_matrix)) ^ 0.5
logit_r_beta_t_ratio      <- abs(logit_r_beta / logit_r_average_std_error)
logit_r_beta_p_value      <- 2 * pnorm(logit_r_beta_t_ratio, mean=0, sd=1, lower.tail=FALSE, log.p=FALSE)
logit_r_log_likelihood    <- (-logit_r_model$aic + 2 * logit_r_model$rank) / 2

summary <- round(cbind(logit_r_beta, logit_r_average_std_error, logit_r_beta_t_ratio, logit_r_beta_p_value), 3)
colnames(summary) <- c("Beta hat", "Średni błąd standardowy", "Statystyka t", "p-value")
rownames(summary) <- c("const", colnames(x_train[, c(1:3, 8:9)]))
summary

# Obliczenie trafności modelu dla zbioru uczącego --------------------------------------------------------------

# Obliczenie wartości przewidywanych (y hat), oznaczających prawdopodobieństwo, że dana osoba przeżyje (y = 1)
x_with_constant <- cbind(1, x_train[, c(1:3, 8:9)])
logit_r_x_t_beta <- x_with_constant %*% logit_r_beta
logit_r_survival_probability <- exp(logit_r_x_t_beta)/(1 + exp(logit_r_x_t_beta))

# Średnia wartość y, na podstawie której stwierdzamy dalej, czy dana osoba przeżyje
p <- colMeans(y_train)
logit_r_y_hat <- as.matrix(as.numeric(logit_r_survival_probability > p))

# Obliczenie tabeli trafności
logit_r_y_comparison <- cbind(y_train, logit_r_y_hat)
logit_r_confusion_matrix <- xtabs(formula = ~ logit_r_y_hat + y_train, data=logit_r_y_comparison)
print("[Model logitowy z restrykcjami] Tabela trafności dla zbioru uczącego:")
print(logit_r_confusion_matrix)

# Obliczenie współczynnika R^2
logit_r_r_squared <- (logit_r_confusion_matrix[1, 1] + logit_r_confusion_matrix[2, 2]) / sum(logit_r_confusion_matrix)
print("[Model logitowy z restrykcjami] Współczynnik R^2 dla zbioru uczącego:")
print(logit_r_r_squared)

# Obliczenie współczynnika determinacji Efrona
logit_r_efron_r_squared <- 1 - (sum((y_train - logit_r_survival_probability) ^ 2) / sum((y_train - mean(y_train)) ^ 2))
print("[Model logitowy z restrykcjami] Współczynnik determinacji Efrona dla zbioru uczącego:")
print(logit_r_efron_r_squared)

# Obliczenie współczynnika determinacji Zavoina i McKelveya
T <- dim(x_train)[1]
T <- T * 3.289868
avg_survival_probability <- mean(logit_r_survival_probability)
nominator <- sum((logit_r_survival_probability - avg_survival_probability) ^ 2)
denominator <- T + nominator
logit_r_zavoin_mckelvey_r_squared <- nominator / denominator
print("[Model logitowy z restrykcjami] Współczynnik determinacji Zavoina i McKelveya dla zbioru uczącego:")
print(logit_r_zavoin_mckelvey_r_squared)

# Efekty krańcowe
logit_r_pdf_values <- pdf_logit(logit_r_x_t_beta)
logit_r_partial_effects <- matrix(0, dim(x_train)[1], ncol(x_with_constant))

for (i in 2:ncol(x_with_constant)) {
  val <- logit_r_pdf_values[, 1] * logit_beta[i]
  logit_r_partial_effects[, i] <- logit_r_pdf_values[, 1] * logit_beta[i]
}

logit_r_mean_partial_effects <- as.matrix(colMeans(logit_r_partial_effects))
colnames(logit_r_mean_partial_effects) <- "[Model logitowy z restrykcjami] Średnie efekty krańcowe dla zbioru uczącego"
rownames(logit_r_mean_partial_effects) <- colnames(x_with_constant)
logit_r_mean_partial_effects

# Obliczenie trafności modelu dla zbioru testowego --------------------------------------------------------------

# Obliczenie wartości przewidywanych (y hat), oznaczających prawdopodobieństwo, że dana osoba przeżyje (y = 1)
x_with_constant <- cbind(1, x_test[, c(1:3, 8:9)])
logit_r_x_t_beta <- x_with_constant %*% logit_r_beta
logit_r_survival_probability <- exp(logit_r_x_t_beta)/(1 + exp(logit_r_x_t_beta))

# Średnia wartość y, na podstawie której stwierdzamy dalej, czy dana osoba przeżyje
p <- colMeans(y_test)
logit_r_y_hat <- as.matrix(as.numeric(logit_r_survival_probability > p))

# Obliczenie tabeli trafności
logit_r_y_comparison <- cbind(y_test, logit_r_y_hat)
logit_r_confusion_matrix <- xtabs(formula = ~ logit_r_y_hat + y_test, data=logit_r_y_comparison)
print("[Model logitowy z restrykcjami] Tabela trafności dla zbioru testowego:")
print(logit_r_confusion_matrix)

# Obliczenie współczynnika R^2
logit_r_r_squared <- (logit_r_confusion_matrix[1, 1] + logit_r_confusion_matrix[2, 2]) / sum(logit_r_confusion_matrix)
print("[Model logitowy z restrykcjami] Współczynnik R^2 dla zbioru testowego:")
print(logit_r_r_squared)

# Obliczenie współczynnika determinacji Efrona
logit_r_efron_r_squared <- 1 - (sum((y_test - logit_r_survival_probability) ^ 2) / sum((y_test - mean(y_test)) ^ 2))
print("[Model logitowy z restrykcjami] Współczynnik determinacji Efrona dla zbioru testowego:")
print(logit_r_efron_r_squared)

# Obliczenie współczynnika determinacji Zavoina i McKelveya
T <- dim(x_test)[1]
T <- T * 3.289868
avg_survival_probability <- mean(logit_r_survival_probability)
nominator <- sum((logit_r_survival_probability - avg_survival_probability) ^ 2)
denominator <- T + nominator
logit_r_zavoin_mckelvey_r_squared <- nominator / denominator
print("[Model logitowy z restrykcjami] Współczynnik determinacji Zavoina i McKelveya dla zbioru testowego:")
print(logit_r_zavoin_mckelvey_r_squared)

# Efekty krańcowe
logit_r_pdf_values <- pdf_logit(logit_r_x_t_beta)
logit_r_partial_effects <- matrix(0, dim(x_test)[1], ncol(x_with_constant))

for (i in 2:ncol(x_with_constant)) {
  logit_r_partial_effects[, i] <- logit_r_pdf_values[, 1] * logit_beta[i]
}

logit_r_mean_partial_effects <- as.matrix(colMeans(logit_r_partial_effects))
colnames(logit_r_mean_partial_effects) <- "[Model logitowy z restrykcjami] Średnie efekty krańcowe dla zbioru testowego"
rownames(logit_r_mean_partial_effects) <- colnames(x_with_constant)
logit_r_mean_partial_effects

# Model logitowy - część 3: Testowanie hipotezy o istotności redukcji modelu
# ----------------------------------------------------------------------------------------------------------------------

# Test LR
# H0: Model bez zmiennych 5:8 jest równie dobry jak model z tymi zmiennymi
# H1: Przynajmniej jedna ze zmiennych 5:8 jest istotna

alpha <- 0.05

logit_lr_test_statistic <- -2 * (logit_r_log_likelihood - logit_log_likelihood)
v <- logit_model$rank - logit_r_model$rank
logit_lr_test_p_value <- pchisq(q=logit_lr_test_statistic, df=v, lower.tail=TRUE, log.p=FALSE)

print("Statystyka testowa:")
print(logit_lr_test_statistic)
print("p-value:")
print(logit_lr_test_p_value)

if (logit_lr_test_p_value < alpha) {
  print("Odrzucamy hipotezę zerową, mówiącą o tym, że model bez zmiennych 5:8 jest równie dobry jak model z tymi zmiennymi")
} else {
  print("Nie ma podstaw do odrzucenia hipotezy zerowej, mówiącej o tym, że model bez zmiennych 5:8 jest równie dobry jak model z tymi zmiennymi")
}

# Model logitowy - część 4: Testowanie hipotez o istotności poszczególnych elementów wektora ocen parametrów
# ----------------------------------------------------------------------------------------------------------------------

alpha <- 0.05
logit_beta_p_value <- 2 * pnorm(logit_beta_t_ratio, mean=0, sd=1, lower.tail=FALSE, log.p=FALSE)

# Zestawione wyniki w formie macierzy o nazwach kolumn "Zmienna", "Statystyka T", "p-value"
logit_beta_test_results <- cbind(logit_beta, logit_beta_t_ratio, logit_beta_p_value)
x_with_constant <- cbind(1, x_train)
colnames(x_with_constant)[1] <- "const"
colnames(logit_beta_test_results) <- c("Zmienna", "Statystyka T", "p-value")
rownames(logit_beta_test_results) <- colnames(x_with_constant)

# Dodanie kolumny z informacją, czy dana zmienna jest istotna
logit_beta_test_results <- cbind(logit_beta_test_results, logit_beta_p_value < alpha)
colnames(logit_beta_test_results)[4] <- "Czy istotna?"

print("[Model Logitowy] Wyniki testów dla poszczególnych elementów wektora ocen parametrów:")
print(logit_beta_test_results)

# Model probitowy - część 1: Oszacowanie modelu probitowego bez restrykcji
# ----------------------------------------------------------------------------------------------------------------------
probit_model <- glm(y_train ~ x_train, family=binomial(link="probit"))
print("Model probitowy bez restrykcji")
print("--------------------------------")

# Obliczenie potrzebnych statystyk dla modelu probitowego
probit_beta              <- as.matrix(probit_model$coefficients)
probit_covariance_matrix <- vcov(probit_model)
probit_average_std_error <- as.matrix(diag(probit_covariance_matrix)) ^ 0.5
probit_beta_t_ratio      <- abs(probit_beta / probit_average_std_error)
probit_beta_p_value      <- 2 * pnorm(probit_beta_t_ratio, mean=0, sd=1, lower.tail=FALSE, log.p=FALSE)
probit_log_likelihood    <- (-probit_model$aic + 2 * probit_model$rank) / 2

summary <- round(cbind(probit_beta, probit_average_std_error, probit_beta_t_ratio, probit_beta_p_value), 3)
colnames(summary) <- c("Beta hat", "Średni błąd standardowy", "Statystyka t", "p-value")
rownames(summary) <- c("const", colnames(x_train))
summary

# Obliczenie trafności modelu dla zbioru uczącego --------------------------------------------------------------

# Obliczenie wartości przewidywanych (y hat), oznaczających prawdopodobieństwo, że dana osoba przeżyje (y = 1)
x_with_constant <- cbind(1, x_train)
probit_x_t_beta <- x_with_constant %*% probit_beta
probit_survival_probability <- pnorm(probit_x_t_beta, mean=0, sd=1, lower.tail=TRUE, log.p=FALSE)

# Średnia wartość y, na podstawie której stwierdzamy dalej, czy dana osoba przeżyje
p <- colMeans(y_train)
probit_y_hat <- as.matrix(as.numeric(probit_survival_probability > p))

# Obliczenie tabeli trafności
probit_y_comparison <- cbind(y_train, probit_y_hat)
probit_confusion_matrix <- xtabs(formula = ~ probit_y_hat + y_train, data=probit_y_comparison)
print("[Model probitowy bez restrykcji] Tabela trafności dla zbioru uczącego:")
print(probit_confusion_matrix)

# Obliczenie współczynnika R^2
probit_r_squared <- (probit_confusion_matrix[1, 1] + probit_confusion_matrix[2, 2]) / sum(probit_confusion_matrix)
print("[Model probitowy bez restrykcji] Współczynnik R^2 dla zbioru uczącego:")
print(probit_r_squared)

# Obliczenie współczynnika determinacji Efrona
probit_efron_r_squared <- 1 - (sum((y_train - probit_survival_probability) ^ 2) / sum((y_train - mean(y_train)) ^ 2))
print("[Model probitowy bez restrykcji] Współczynnik determinacji Efrona dla zbioru uczącego:")
print(probit_efron_r_squared)

# Obliczenie współczynnika deterinacji Zavoina i McKelveya
T <- dim(x_train)[1]
avg_survival_probability <- mean(probit_survival_probability)
nominator <- sum((probit_survival_probability - avg_survival_probability) ^ 2)
denominator <- T + nominator
probit_zavoin_mckelvey_r_squared <- nominator / denominator
print("[Model probitowy bez restrykcji] Współczynnik determinacji Zavoina i McKelveya dla zbioru uczącego:")
print(probit_zavoin_mckelvey_r_squared)

# Efekty krańcowe
probit_pdf_values <- pdf_probit(probit_x_t_beta)
probit_partial_effects <- matrix(0, dim(x_train)[1], ncol(x_with_constant))

for (i in 2:ncol(x_with_constant)) {
  probit_partial_effects[, i] <- probit_pdf_values[, 1] * probit_beta[i]
}

probit_mean_partial_effects <- as.matrix(colMeans(probit_partial_effects))
colnames(probit_mean_partial_effects) <- "[Model probitowy bez restrykcji] Średnie efekty krańcowe dla zbioru uczącego"
rownames(probit_mean_partial_effects) <- colnames(x_with_constant)
probit_mean_partial_effects

# Obliczenie trafności modelu dla zbioru testowego --------------------------------------------------------------

# Obliczenie wartości przewidywanych (y hat), oznaczających prawdopodobieństwo, że dana osoba przeżyje (y = 1)
x_with_constant <- cbind(1, x_test)
probit_x_t_beta <- x_with_constant %*% probit_beta
probit_survival_probability <- pnorm(probit_x_t_beta, mean=0, sd=1, lower.tail=TRUE, log.p=FALSE)

# Średnia wartość y, na podstawie której stwierdzamy dalej, czy dana osoba przeżyje
p <- colMeans(y_test)
probit_y_hat <- as.matrix(as.numeric(probit_survival_probability > p))

# Obliczenie tabeli trafności
probit_y_comparison <- cbind(y_test, probit_y_hat)
probit_confusion_matrix <- xtabs(formula = ~ probit_y_hat + y_test, data=probit_y_comparison)
print("[Model probitowy bez restrykcji] Tabela trafności dla zbioru testowego:")
print(probit_confusion_matrix)

# Obliczenie współczynnika R^2
probit_r_squared <- (probit_confusion_matrix[1, 1] + probit_confusion_matrix[2, 2]) / sum(probit_confusion_matrix)
print("[Model probitowy bez restrykcji] Współczynnik R^2 dla zbioru testowego:")
print(probit_r_squared)

# Obliczenie współczynnika deterinacji Efrona
probit_efron_r_squared <- 1 - (sum((y_test - probit_survival_probability) ^ 2) / sum((y_test - mean(y_test)) ^ 2))
print("[Model probitowy bez restrykcji] Współczynnik determinacji Efrona dla zbioru testowego:")
print(probit_efron_r_squared)

# Obliczenie współczynnika deterinacji Zavoina i McKelveya
T <- dim(x_test)[1]
avg_survival_probability <- mean(probit_survival_probability)
nominator <- sum((probit_survival_probability - avg_survival_probability) ^ 2)
denominator <- T + nominator
probit_zavoin_mckelvey_r_squared <- nominator / denominator
print("[Model probitowy bez restrykcji] Współczynnik determinacji Zavoina i McKelveya dla zbioru testowego:")
print(probit_zavoin_mckelvey_r_squared)

# Efekty krańcowe
probit_pdf_values <- pdf_probit(probit_x_t_beta)
probit_partial_effects <- matrix(0, dim(x_test)[1], ncol(x_with_constant))

for (i in 2:ncol(x_with_constant)) {
  probit_partial_effects[, i] <- probit_pdf_values[, 1] * probit_beta[i]
}

probit_mean_partial_effects <- as.matrix(colMeans(probit_partial_effects))
colnames(probit_mean_partial_effects) <- "[Model probitowy bez restrykcji] Średnie efekty krańcowe dla zbioru testowego"
rownames(probit_mean_partial_effects) <- colnames(x_with_constant)
probit_mean_partial_effects

# Model probitowy - część 2: Oszacowanie modelu probitowego z restrykcjami na kolumnach 5:8
# ----------------------------------------------------------------------------------------------------------------------
probit_r_model <- glm(y_train ~ x_train[, c(1:3, 8:9)], family=binomial(link="probit"))
print("Model probitowy z restrykcjami na kolumnach 5:8")
print("-----------------------------------------------")

# Obliczenie potrzebnych statystyk dla modelu probitowego
probit_r_beta              <- as.matrix(probit_r_model$coefficients)
probit_r_covariance_matrix <- vcov(probit_r_model)
probit_r_average_std_error <- as.matrix(diag(probit_r_covariance_matrix)) ^ 0.5
probit_r_beta_t_ratio      <- abs(probit_r_beta / probit_r_average_std_error)
probit_r_beta_p_value      <- 2 * pnorm(probit_r_beta_t_ratio, mean=0, sd=1, lower.tail=FALSE, log.p=FALSE)
probit_r_log_likelihood    <- (-probit_r_model$aic + 2 * probit_r_model$rank) / 2

summary <- round(cbind(probit_r_beta, probit_r_average_std_error, probit_r_beta_t_ratio, probit_r_beta_p_value), 3)
colnames(summary) <- c("Beta hat", "Średni błąd standardowy", "Statystyka t", "p-value")
rownames(summary) <- c("const", colnames(x_train[, c(1:3, 8:9)]))
summary

# Obliczenie trafności modelu dla zbioru uczącego --------------------------------------------------------------

# Obliczenie wartości przewidywanych (y hat), oznaczających prawdopodobieństwo, że dana osoba przeżyje (y = 1)
x_with_constant <- cbind(1, x_train[, c(1:3, 8:9)])
probit_r_x_t_beta <- x_with_constant %*% probit_r_beta
probit_r_survival_probability <- pnorm(probit_r_x_t_beta, mean=0, sd=1, lower.tail=TRUE, log.p=FALSE)

# Średnia wartość y, na podstawie której stwierdzamy dalej, czy dana osoba przeżyje
p <- colMeans(y_train)
probit_r_y_hat <- as.matrix(as.numeric(probit_r_survival_probability > p))

# Obliczenie tabeli trafności
probit_r_y_comparison <- cbind(y_train, probit_r_y_hat)
probit_r_confusion_matrix <- xtabs(formula = ~ probit_r_y_hat + y_train, data=probit_r_y_comparison)
print("[Model probitowy z restrykcjami] Tabela trafności dla zbioru uczącego:")
print(probit_r_confusion_matrix)

# Obliczenie współczynnika R^2
probit_r_r_squared <- (probit_r_confusion_matrix[1, 1] + probit_r_confusion_matrix[2, 2]) / sum(probit_r_confusion_matrix)
print("[Model probitowy z restrykcjami] Współczynnik R^2 dla zbioru uczącego:")
print(probit_r_r_squared)

# Obliczenie współczynnika determinacji Efrona
probit_r_efron_r_squared <- 1 - (sum((y_train - probit_r_survival_probability) ^ 2) / sum((y_train - mean(y_train)) ^ 2))
print("[Model probitowy z restrykcjami] Współczynnik determinacji Efrona dla zbioru uczącego:")
print(probit_r_efron_r_squared)

# Obliczenie współczynnika determinacji Zavoina i McKelveya
T <- dim(x_train)[1]
avg_survival_probability <- mean(probit_r_survival_probability)
nominator <- sum((probit_r_survival_probability - avg_survival_probability) ^ 2)
denominator <- T + nominator
probit_r_zavoin_mckelvey_r_squared <- nominator / denominator
print("[Model probitowy z restrykcjami] Współczynnik determinacji Zavoina i McKelveya dla zbioru uczącego:")
print(probit_r_zavoin_mckelvey_r_squared)

# Efekty krańcowe
probit_r_pdf_values <- pdf_probit(probit_r_x_t_beta)
probit_r_partial_effects <- matrix(0, dim(x_train)[1], ncol(x_with_constant))

for (i in 2:ncol(x_with_constant)) {
  probit_r_partial_effects[, i] <- probit_r_pdf_values[, 1] * probit_r_beta[i]
}

probit_r_mean_partial_effects <- as.matrix(colMeans(probit_r_partial_effects))
colnames(probit_r_mean_partial_effects) <- "[Model probitowy z restrykcjami] Średnie efekty krańcowe dla zbioru uczącego"
rownames(probit_r_mean_partial_effects) <- colnames(x_with_constant)
probit_r_mean_partial_effects

# Obliczenie trafności modelu dla zbioru testowego --------------------------------------------------------------

# Obliczenie wartości przewidywanych (y hat), oznaczających prawdopodobieństwo, że dana osoba przeżyje (y = 1)
x_with_constant <- cbind(1, x_test[, c(1:3, 8:9)])
probit_r_x_t_beta <- x_with_constant %*% probit_r_beta
probit_r_survival_probability <- pnorm(probit_r_x_t_beta, mean=0, sd=1, lower.tail=TRUE, log.p=FALSE)

# Średnia wartość y, na podstawie której stwierdzamy dalej, czy dana osoba przeżyje
p <- colMeans(y_test)
probit_r_y_hat <- as.matrix(as.numeric(probit_r_survival_probability > p))

# Obliczenie tabeli trafności
probit_r_y_comparison <- cbind(y_test, probit_r_y_hat)
probit_r_confusion_matrix <- xtabs(formula = ~ probit_r_y_hat + y_test, data=probit_r_y_comparison)
print("[Model probitowy z restrykcjami] Tabela trafności dla zbioru testowego:")
print(probit_r_confusion_matrix)

# Obliczenie współczynnika R^2
probit_r_r_squared <- (probit_r_confusion_matrix[1, 1] + probit_r_confusion_matrix[2, 2]) / sum(probit_r_confusion_matrix)
print("[Model probitowy z restrykcjami] Współczynnik R^2 dla zbioru testowego:")
print(probit_r_r_squared)

# Obliczenie współczynnika determinacji Efrona
probit_r_efron_r_squared <- 1 - (sum((y_test - probit_r_survival_probability) ^ 2) / sum((y_test - mean(y_test)) ^ 2))
print("[Model probitowy z restrykcjami] Współczynnik determinacji Efrona dla zbioru testowego:")
print(probit_r_efron_r_squared)

# Obliczenie współczynnika determinacji Zavoina i McKelveya
T <- dim(x_test)[1]
avg_survival_probability <- mean(probit_r_survival_probability)
nominator <- sum((probit_r_survival_probability - avg_survival_probability) ^ 2)
denominator <- T + nominator
probit_r_zavoin_mckelvey_r_squared <- nominator / denominator
print("[Model probitowy z restrykcjami] Współczynnik determinacji Zavoina i McKelveya dla zbioru testowego:")
print(probit_r_zavoin_mckelvey_r_squared)

# Efekty krańcowe
probit_r_pdf_values <- pdf_probit(probit_r_x_t_beta)
probit_r_partial_effects <- matrix(0, dim(x_test)[1], ncol(x_with_constant))

for (i in 2:ncol(x_with_constant)) {
  probit_r_partial_effects[, i] <- probit_r_pdf_values[, 1] * probit_r_beta[i]
}

probit_r_mean_partial_effects <- as.matrix(colMeans(probit_r_partial_effects))
colnames(probit_r_mean_partial_effects) <- "[Model probitowy z restrykcjami] Średnie efekty krańcowe dla zbioru testowego"
rownames(probit_r_mean_partial_effects) <- colnames(x_with_constant)
probit_r_mean_partial_effects

# Model probitowy - część 3: Testowanie hipotezy o istotności redukcji modelu
# ----------------------------------------------------------------------------------------------------------------------

# Test LR
# H0: Model bez zmiennych 5:8 jest równie dobry jak model z tymi zmiennymi
# H1: Przynajmniej jedna ze zmiennych 5:8 jest istotna

alpha <- 0.05

probit_lr_test_statistic <- -2 * (probit_r_log_likelihood - probit_log_likelihood)
v <- probit_model$rank - probit_r_model$rank
probit_lr_test_p_value <- pchisq(q=probit_lr_test_statistic, df=v, lower.tail=TRUE, log.p=FALSE)

print("Statystyka testowa:")
print(probit_lr_test_statistic)
print("p-value:")
print(probit_lr_test_p_value)

if (probit_lr_test_p_value < alpha) {
  print("Odrzucamy hipotezę zerową, mówiącą o tym, że model bez zmiennych 5:8 jest równie dobry jak model z tymi zmiennymi")
} else {
  print("Nie ma podstaw do odrzucenia hipotezy zerowej, mówiącej o tym, że model bez zmiennych 5:8 jest równie dobry jak model z tymi zmiennymi")
}

# Model probitowy - część 4: Testowanie hipotez o istotności poszczególnych elementów wektora ocen parametrów
# ----------------------------------------------------------------------------------------------------------------------

alpha <- 0.05
probit_beta_p_value <- 2 * pnorm(probit_beta_t_ratio, mean=0, sd=1, lower.tail=FALSE, log.p=FALSE)

# Zestawione wyniki w formie macierzy o nazwach kolumn "Zmienna", "Statystyka T", "p-value"
probit_beta_test_results <- cbind(probit_beta, probit_beta_t_ratio, probit_beta_p_value)
x_with_constant <- cbind(1, x_train)
colnames(x_with_constant)[1] <- "const"
colnames(probit_beta_test_results) <- c("Zmienna", "Statystyka T", "p-value")
rownames(probit_beta_test_results) <- colnames(x_with_constant)

# Dodanie kolumny z informacją, czy dana zmienna jest istotna
probit_beta_test_results <- cbind(probit_beta_test_results, probit_beta_p_value < alpha)
colnames(probit_beta_test_results)[4] <- "Czy istotna?"

print("[Model probitowy] Wyniki testów dla poszczególnych elementów wektora ocen parametrów:")
print(probit_beta_test_results)
