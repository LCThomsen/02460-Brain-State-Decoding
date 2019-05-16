library(dplyr)
library(CCA)
library(Rtsne)
library(FactoMineR)
library(ggplot2)
library(factoextra)
library(FactoMineR)

library(CCP)
library(corrplot)
library(GGally)
## == load data ==  ## 
EEG <- read.table("C:/Users/kaspe/OneDrive/DTU/12. Semester/Advanced machine learning/Mindreader folder/data/exp4/X_5000.txt")

SEM <- read.table("D:/Advanced machine learning/Mindreader folder/data/exp1/semantics.txt")

cat <- read.table("D:/Advanced_v2/Advanced machine learning/Mindreader folder/data/exp4/category.txt")
#SEM_scale <- scale(SEM)
tsne <- Rtsne(SEM, dims = 2, perplexity=60, verbose=TRUE, max_iter = 700,check_duplicates=F)
tsne <- Rtsne(SEM_scale, dims = 2, perplexity=50, verbose=TRUE, max_iter = 700,check_duplicates=F)
KM <- kmeans(SEM_scale,centers = 6,nstart=25)
tsne_data <- data.frame(tsne$Y,category =cat)
names(tsne_data)
ggplot(data = tsne_data,aes(x = X1,y=X2,col=as.factor(V1))) + geom_point()


## regularized CCA due to large amount of parameters
#RCC <- rcc(EEG,SEM,lambda1 = 0.1,lambda2 = 0.5)
#RCC1 <- rcc(EEG,SEM,lambda1 = 20,lambda2 = 4)

#saveRDS(RCC,"RCC_5000.rds")
RCC <- readRDS("D:/Advanced_v2/Advanced machine learning/RCC.rds")

plt.cc(RCC,type="v")
legend("topright",c("EEG","Image semantics"),col=c("blue","red"),pch=c(2,2),bty = "o",cex = 0.8)

library(scales)
par(mfrow=c(1,2))
plot(1, type="n", xlab="Figure (1)    Canonical var 1", ylab="Canonical var 2", xlim=c(-1, 1), ylim=c(-1, 1),main = "Variables plotted on (1st,2nd) canonical variate")
points(RCC$scores$corr.X.xscores[,1],RCC$scores$corr.X.xscores[,2], cex= 0.7,col="red",pch=2)
points(RCC$scores$corr.Y.xscores[,1],RCC$scores$corr.Y.xscores[,2], cex= 0.7,col=alpha("blue",0.5),pch=2)
abline(h=0,lty=2)
abline(v=0,lty=2)
legend(c(0,-1.25),c(-2,-0.25),c("Image semantics","EEG"),col=c("blue","red"),pch=c(2,2),bty = "n",cex=1,pt.cex=3)

Plot_data <- data.frame(Dim1 = RCC$scores$xscores[,1],Dim2 = RCC$scores$xscores[,2],Category = cat)
colnames(Plot_data) <- c("Dim1","Dim2","Category")

plot(1, type="n", xlab="Figure(2)     Canonical var 1", ylab=" Canonical var 2", xlim=c(-3, 2), ylim=c(-2, 3),main = "Individual obs. plotted on (1st,2nd) canonical variate")
points(Plot_data$Dim1,Plot_data$Dim2, cex= 0.7,col=Plot_data$Category,pch=2)
abline(h=0,lty=2)
abline(v=0,lty=2)


ggplot(Plot_data,aes(x=Dim1,y=Dim2,col=Category)) + geom_point() + ggtitle("Individual obs. plotted onto canonical variates") + theme_bw()

rho_rcc <- RCC$cor
## Calculate p-values using the F-approximations of different test statistics:
options(scipen=999)
n <- 2160
p <- 2048
q <- 2048
## Calculate p-values using the F-approximations of different test statistics:
p.asym(rho_rcc, n, p, q, tstat = "Wilks")
warnings()p.asym(rho_rcc, n, p, q, tstat = "Hotelling")
p.asym(rho_rcc, n, p, q, tstat = "Pillai")



######### ================  PCA NPAIRS  RESAMPLING TEST 
library(Morpho)
X <- read.table("D:/Advanced_v2/Advanced machine learning/Mindreader folder/data/exp4/X_2600.txt")

#X1 <- read.table("D:/Advanced_v2/Advanced machine learning/Mindreader folder/data/exp4/X1_pca.txt")
#X2 <- read.table("D:/Advanced_v2/Advanced machine learning/Mindreader folder/data/exp4/X2_pca.txt")
length1 = 50
hsu1 <- matrix(NA,nrow=length1)
hsu2 <- matrix(NA,nrow=length1)
hsu3 <- matrix(NA,nrow=length1)
hsu4 <- matrix(NA,nrow=length1)
hsu5 <- matrix(NA,nrow=length1)

for ( i in 1:length1){
idx <- sample(1:2160,1080)
X1 <- X[idx,]
X2 <- X[-idx,]
PCA1 <- PCA(X1,scale.unit = F,ncp = 300,graph = F)
PCA2 <- PCA(X2,scale.unit = F,ncp = 300,graph = F)
hsu1[i] <- cos(angle.calc(PCA1$var$coord[,1],PCA2$var$coord[,1]))
hsu2[i] <- cos(angle.calc(PCA1$var$coord[,2],PCA2$var$coord[,2]))
hsu3[i] <- cos(angle.calc(PCA1$var$coord[,3],PCA2$var$coord[,3]))
hsu4[i] <- cos(angle.calc(PCA1$var$coord[,4],PCA2$var$coord[,4]))
hsu5[i] <- cos(angle.calc(PCA1$var$coord[,5],PCA2$var$coord[,5]))

print(i)

}


plot(abs(hsu1),pch=20,main="Absolute cosine value of angle respective PCs" ,ylab="Cosine to angle between PCs",ylim=c(0.3,1),xlab="Iteration")
abline(h = mean(abs(hsu1)),lty=2)

abline(h = mean(abs(hsu2)),lty=2,col="red")
abline(h = mean(abs(hsu3)),lty=2,col="blue")
abline(h = mean(abs(hsu4)),lty=2,col="orange")
abline(h = mean(abs(hsu5)),lty=2,col="green")


legend("bottomright",legend=c("mean for PC1s","mean for PC2s","mean for PC3s","mean for PC4s","mean for PC5s"),lty=2,col=c("black","red","blue","orange","green"))

#plot(ft,type="l",xlab="Prinicap component",ylab="Angle (degrees)",main = "Degrees between pairwise eigenvectors of split data")


t.test(abs(hsu1))
t.test(abs(hsu5))



idx <- sample(1:2160,1080)
X1 <- X[idx,]
X2 <- X[-idx,]
PCA1 <- prcomp(X1,center=F,scale. = F)
PCA2 <- prcomp(X2,center=F,scale. = F)
angle.calc(PCA1$x[,1],PCA2$x[,1])
