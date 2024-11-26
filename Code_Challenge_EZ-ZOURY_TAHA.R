# Majeure Science des données 2023-2024
# UP2 - Apprentissage statistique
# TP Challenge
# Taha EZ-ZOURY

rm(list=ls())
set.seed(1234)

# Q0 - Chargement des données d'apprentissage, puis on la divise en données train et test
data <- read.table(file="data.txt", header=TRUE)
n <- dim(data)[1]
p <- dim(data)[2] - 1
n.app <- 0.8*n
index_app <- sample(1:n, n.app, replace=FALSE)
data.app <- data[index_app,]
data.test <- data[-index_app,]
X.app <- as.matrix(data.app[,2:(p+1)])
y.app <- data.app[,1]


# Q1 - RMSE de référence
RMSE_ref=sd(data[,1])
print(RMSE_ref)
print(sd(data.test[,1]))


# Q2 - Régression linéaire multiple (MCO)
reg <- lm(y.app ~ X.app )   
summary(reg) 
# Donc la regression linéaire multiple ne marche pas pour cette regression car p>n .


# Modèle LASSO
library(glmnet)
lasso <- glmnet(X.app, y.app)
# Sélection du paramètre de régularisation lambda
set.seed(1234)
Llasso <- cv.glmnet(X.app, y.app)
lambda.lasso <- Llasso$lambda.min
plot(Llasso)
# Prédictions
X.test <- as.matrix(data.test[,2:(p+1)])
y.test <- data.test[,1]
pred.lasso <- predict(Llasso, newx=X.test, s=lambda.lasso)
plot(y.test ~ pred.lasso, pch=1)
abline(a=0,b=1,col='red',lwd=2)
n.test <- n - n.app
RMSE.lasso <- sqrt( sum( (y.test - pred.lasso)^2 )/n.test )
print(RMSE.lasso)
print(RMSE_ref)


# Comparant avec RIDGE: 
ridge <- glmnet(X.app, y.app, alpha = 0)
Lridge <- cv.glmnet(X.app, y.app, alpha=0)
lambda.ridge <- Lridge$lambda.min
pred.ridge <- predict(Lridge,newx=X.test, alpha=0, s=lambda.ridge)
RMSE.ridge <- sqrt( sum( (y.test - pred.ridge)^2 )/n.test )
print(RMSE.ridge)

# RMSE.ridge = 6.1207, Donc le modèle lasso est mieux que RIDGE, on essaiera d'améliorer le modèle lasso.

#Améliorisation 1: Normalisation des données - choix de lambda par 20 fold cv  
X.app_scaled <- scale(X.app)

# Effectuer une validation croisée plus rigoureuse pour lambda (20 fold cv)
set.seed(1234)
Llasso_amel1 <- cv.glmnet(X.app_scaled, y.app, nfolds = 20, alpha = 1)  
lambda_lasso_amel1 <- Llasso_amel1$lambda.min
# Prédictions
pred.lasso.amel1 <- predict(Llasso_amel1, newx=X.test, s=lambda_lasso_amel1)
plot(y.test ~ pred.lasso.amel1, pch=1)
abline(a=0,b=1,col='red',lwd=2)
RMSE.lasso.amel1 <- sqrt( sum( (y.test - pred.lasso.amel1)^2 )/n.test )
print(RMSE.lasso.amel1)
print(RMSE.lasso)

# le RMSE a été bien amélioré (de 5.08 à 4.76)


# Amélioration 2: réduire la dimension des données train par l'ACP

# Effectuer l'Analyse en Composantes Principales (ACP)
pca_model <- prcomp(X.app, scale = TRUE)
# Choisir le nombre de composantes principales à conserver en conservant 90% de l'inertie totale
variance_explained <- cumsum(pca_model$sdev^2) / sum(pca_model$sdev^2)
num_components <- which.max(variance_explained > 0.90)
num_components  #On peut remarquer qu'on a réduit la dimension de 200 à 51 !!
# Réduire les données d'apprentissage aux composantes principales sélectionnées
X.app_pca <- pca_model$x[, 1:num_components]
# Effectuer une validation croisée pour lambda avec les données réduites par ACP
set.seed(1234)
Llasso_pca <- cv.glmnet(X.app_pca, y.app, nfolds = 20, alpha = 1)
lambda_lasso_pca <- Llasso_pca$lambda.min
# Réduire les données de test aux mêmes composantes principales
X.test_pca <- predict(pca_model, newdata = X.test)[, 1:num_components]
# Prédictions avec les données réduites par ACP
pred.lasso_pca <- predict(Llasso_pca, newx = X.test_pca, s = lambda_lasso_pca)

# Calculer RMSE pour le modèle avec les données réduites par ACP
RMSE_lasso_pca <- sqrt(sum((y.test - pred.lasso_pca)^2) / n.test)
print(RMSE_lasso_pca)

#On peut voir que notre erreur a augmenté car les premières composantes principales 
#ne sont pas forcément les variables les  plus corrélées avec la réponse Y, 
#Après une recherche d'amélioration, on peut améliorer ça en utilisant la régression par 
#moindres carrés partiels (pls).


# Amélioration 3: essayant le modèle pls

library(pls)
# Effectuer PLSR sur les données d'apprentissage réduites par ACP
plsr_model <- plsr(y.app ~ X.app_pca, scale = TRUE, validation = "CV")
# Extraire les prédictions du modèle PLSR pour les différentes composantes
pred_cv <- predict(plsr_model)
# Calculer le RMSE pour chaque composante et sélectionner le nombre optimal de composantes
RMSE_cv <- apply(pred_cv, 2, function(pred) sqrt(mean((y.app - pred)^2)))
num_components_plsr <- which.min(RMSE_cv)
# Entraîner le modèle PLSR avec le nombre optimal de composantes
plsr_model_final <- plsr(y.app ~ X.app_pca, ncomp = num_components_plsr, scale = TRUE)

# Prédire les valeurs pour les données de test réduites par ACP
pred_plsr <- predict(plsr_model_final, newdata = X.test_pca)

# Calculer RMSE pour le modèle PLSR avec les données réduites par ACP
RMSE_plsr <- sqrt(sum((y.test - pred_plsr)^2) / length(y.test))

print(RMSE_plsr)
print(RMSE_lasso_pca)
print(RMSE.lasso.amel1)
print(RMSE.lasso)
print(RMSE_ref)

# Le dernier modèle pls est mieux que celui d'avant (par l'ACP) mais le modèle lasso.amel1 
#reste le meilleure modèle en terme de minimalisation de RMSE. On prendra alors ce modèle sans 
#continuer sur d'autres améliorations sous risque de tomber dans le sur-appretissage.



# Q5 - Chargement des données test
X.data.test <- read.table(file="Xtest.txt", header=TRUE)
Xtest <- as.matrix(X.data.test)

# Prédictions
pred <- predict(Llasso_amel1, newx=Xtest, s=lambda_lasso_amel1)
pred
summary(pred)
summary(data[,1])
sd(pred)
sd(data[,1])

# Enregistrer les prédictions
write.table(pred, file = "EZ-ZOURY.txt", row.names = FALSE, col.names = FALSE)





