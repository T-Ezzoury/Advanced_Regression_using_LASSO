# Majeure Science des données 2022-2023
# UP2 - Apprentissage statistique
# TP sur la Régression LASSO

# Chargement des données
data <- read.table(file="data_LASSO.txt", header=TRUE)
names(data)
pairs(data)
n <- dim(data)[1]
p <- dim(data)[2] - 1
n.app <- 0.8*n

index_app <- sample(1:n, n.app, replace=FALSE)
data.app <- data[index_app,]
data.test <- data[-index_app,]

X.app <- as.matrix(data.app[,2:(p+1)])
y.app <- data.app[,1]

# Modèle LASSO
library(glmnet)
lasso <- glmnet(X.app, y.app)

# Modèle RIDGE
ridge <- glmnet(X.app, y.app, alpha = 0)

# Chemins de régularisation
par(mfrow=c(1,2))
plot(lasso, xvar="lambda"); grid()
plot(ridge, xvar="lambda"); grid()
par(mfrow=c(1,1))

# On modifie la plage des valeurs de lambda
# lambda <- exp(seq(-20,1,0.5))
# lasso <- glmnet(X.app, y.app, lambda=lambda)
# ridge <- glmnet(X.app, y.app, alpha = 0, lambda=lambda)
# 
# par(mfrow=c(1,2))
# plot(lasso, xvar="lambda"); grid()
# plot(ridge, xvar="lambda"); grid()
# par(mfrow=c(1,1))

# Sélection du paramètre de régularisation lambda
set.seed(2022)
Llasso <- cv.glmnet(X.app, y.app)
lambda.lasso <- Llasso$lambda.min

Lridge <- cv.glmnet(X.app, y.app, alpha=0)
lambda.ridge <- Lridge$lambda.min

par(mfrow=c(1,2))
plot(Llasso)
plot(Lridge)
par(mfrow=c(1,1))

# Prédictions

X.test <- as.matrix(data.test[,2:(p+1)])

pred.lasso <- predict(Llasso, newx=X.test, s=lambda.lasso)
pred.ridge <- predict(Lridge,newx=X.test, alpha=0, s=lambda.ridge)

y.test <- data.test[,1]

par(mfrow=c(1,2))
plot(y.test ~ pred.lasso, pch=1)
abline(a=0,b=1,col='red',lwd=2)
plot(y.test ~ pred.ridge, pch=1)
abline(a=0,b=1,col='red',lwd=2)
par(mfrow=c(1,1))

n.test <- n - n.app
RMSE.lasso <- sqrt( sum( (y.test - pred.lasso)^2 )/n.test )
RMSE.ridge <- sqrt( sum( (y.test - pred.ridge)^2 )/n.test )
print(RMSE.lasso)
print(RMSE.ridge)
print(sd(y.test))


