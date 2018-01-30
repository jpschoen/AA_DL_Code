


# clear workspace, set seed, and set wd
rm(list=ls())
set.seed(19)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#read in data
df <- read.csv("data_clean/train_1.csv", stringsAsFactors=FALSE)
colnames(df)
#subset if city and state don't match
df_sub <- subset(df, df$ID1_city!=df$ID2_city & df$ID1_state!=df$ID2_state)
