#=============================================================#
# John Schoeneman
# Work Done For: HT Project, OSU
# Date: Summer 2017
# Work Done: Count data
# Machine: MacPro OSX Yosemite
#=============================================================#


# clear workspace, set seed, and set wd
rm(list=ls())
set.seed(19)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


# load libraries

# load matches data
matches <- read.csv("matches_no", stringsAsFactors=FALSE) 
matches <- na.omit(matches)
sum(matches)

# load matched data
matches1 <- read.csv("data_matched/p_match_10k", stringsAsFactors=FALSE) 
matches2 <- read.csv("data_matched/p_match_20k", stringsAsFactors=FALSE) 
matches3 <- read.csv("data_matched/p_match_30k", stringsAsFactors=FALSE) 
matches4 <- read.csv("data_matched/p_match_40k", stringsAsFactors=FALSE) 
matches5 <- read.csv("data_matched/p_match_50k", stringsAsFactors=FALSE) 

matched <- length(matches1[,1]) + length(matches2[,1]) + length(matches3[,1]) + 
             length(matches4[,1]) + length(matches5[,1])
matched/sum(matches) # 0.5629903, 6.3million


df <- matches1
#order for splitting
df <- df[,-1]
df <- df[,c(2,3,4,5,6,7,8,1,9,10,11,12,13,14)]
#split, randomly sort, and bind back
df$sorter <- rnorm(length(df [,1]),0,10)
h1 <- df[,1:7]
h2 <- df[,8:15]
h2 <- h2[order(h2$sorter),]
df <- cbind(h1, h2)
rm(h1,h2)
#drop non-matches
df$matched <- ifelse(df$ID1_phone==df$ID2_phone,1,0)
df <- subset(df, df$matched==0)
df <- df[,-15]
#add to good matches
df2 <- matches1
#order for splitting
df2 <- df2[,-1]
df2 <- df2[,c(2,3,4,5,6,7,8,1,9,10,11,12,13,14,17)]
df <- rbind(df,df2)
rm(df2)
#save new files

write.csv(df, file = "train_1")



df <- matches2
#order for splitting
df <- df[,-1]
df <- df[,c(2,3,4,5,6,7,8,1,9,10,11,12,13,14)]
#split, randomly sort, and bind back
df$sorter <- rnorm(length(df [,1]),0,10)
h1 <- df[,1:7]
h2 <- df[,8:15]
h2 <- h2[order(h2$sorter),]
df <- cbind(h1, h2)
rm(h1,h2)
#drop non-matches
df$matched <- ifelse(df$ID1_phone==df$ID2_phone,1,0)
df <- subset(df, df$matched==0)
df <- df[,-15]
#add to good matches
df2 <- matches2
#order for splitting
df2 <- df2[,-1]
df2 <- df2[,c(2,3,4,5,6,7,8,1,9,10,11,12,13,14,17)]
df <- rbind(df,df2)
rm(df2)
#save new files

write.csv(df, file = "train_2")

df <- matches3
#order for splitting
df <- df[,-1]
df <- df[,c(2,3,4,5,6,7,8,1,9,10,11,12,13,14)]
#split, randomly sort, and bind back
df$sorter <- rnorm(length(df [,1]),0,10)
h1 <- df[,1:7]
h2 <- df[,8:15]
h2 <- h2[order(h2$sorter),]
df <- cbind(h1, h2)
rm(h1,h2)
#drop non-matches
df$matched <- ifelse(df$ID1_phone==df$ID2_phone,1,0)
df <- subset(df, df$matched==0)
df <- df[,-15]
#add to good matches
df2 <- matches3
#order for splitting
df2 <- df2[,-1]
df2 <- df2[,c(2,3,4,5,6,7,8,1,9,10,11,12,13,14,17)]
df <- rbind(df,df2)
rm(df2)
#save new files

write.csv(df, file = "train_3")


df <- matches4
#order for splitting
df <- df[,-1]
df <- df[,c(2,3,4,5,6,7,8,1,9,10,11,12,13,14)]
#split, randomly sort, and bind back
df$sorter <- rnorm(length(df [,1]),0,10)
h1 <- df[,1:7]
h2 <- df[,8:15]
h2 <- h2[order(h2$sorter),]
df <- cbind(h1, h2)
rm(h1,h2)
#drop non-matches
df$matched <- ifelse(df$ID1_phone==df$ID2_phone,1,0)
df <- subset(df, df$matched==0)
df <- df[,-15]
#add to good matches
df2 <- matches4
#order for splitting
df2 <- df2[,-1]
df2 <- df2[,c(2,3,4,5,6,7,8,1,9,10,11,12,13,14,17)]
df <- rbind(df,df2)
rm(df2)
#save new files

write.csv(df, file = "train_4")


df <- matches5
#order for splitting
df <- df[,-1]
df <- df[,c(2,3,4,5,6,7,8,1,9,10,11,12,13,14)]
#split, randomly sort, and bind back
df$sorter <- rnorm(length(df [,1]),0,10)
h1 <- df[,1:7]
h2 <- df[,8:15]
h2 <- h2[order(h2$sorter),]
df <- cbind(h1, h2)
rm(h1,h2)
#drop non-matches
df$matched <- ifelse(df$ID1_phone==df$ID2_phone,1,0)
df <- subset(df, df$matched==0)
df <- df[,-15]
#add to good matches
df2 <- matches5
#order for splitting
df2 <- df2[,-1]
df2 <- df2[,c(2,3,4,5,6,7,8,1,9,10,11,12,13,14,17)]
df <- rbind(df,df2)
rm(df2)
#save new files

write.csv(df, file = "train_5")



