#=============================================================#
# John Schoeneman
# Work Done For: HT Project, OSU
# Date: FA17
# Work Done: subset data
#=============================================================#


# clear workspace, set seed, and set wd
rm(list=ls())
set.seed(19)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


# Packages
library(textreuse)
library(stringr)
library(foreach)
library(doSNOW)

#assign cores for parallel
#cl <- makeCluster(8, "SOCK") # where 3 is the number of cores used
#registerDoSNOW(cl)

#Data Folder
path <- "data_clean/"

for(j in 1:4){
  # load data
  start.time <- Sys.time()
  df <- read.csv(paste0(path,"train_", j, ".csv"), stringsAsFactors=FALSE)
  
  #subset by freq
  df <- df[df$ID1_freq>5 & df$ID1_freq<617 & df$ID2_freq>5 & df$ID2_freq<617,]
  
  # Apply Smith Waterman alogrithm score 
  # as percentage of post one
  df$sw_score <- 0
  df$post1_length <- 0
  options(warn=-1)
  for(i in 1:nrow(df)){
    df$post1_length[i] <-str_count(df$ID1_post[i], "\\S+")
    a <- tryCatch(
      align_local(df$ID1_post[i], df$ID2_post[i], gap =0)[3],
      error=function(e) e)
    if(inherits(a, "error")) next
    
    #REAL WORK
    df$sw_score[i] <- as.numeric(align_local(df$ID1_post[i], df$ID2_post[i], gap =0)[3])/df$post1_length[i]
    
  }
  options(warn=0)
  
  
  #subset by score and post length
  df <- df[df$sw_score<1.5,]
  
  write.csv(df, file = paste0(path,"ht_subset_", j, ".csv"))
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  print(time.taken)

}

# clear workspace, set seed, and set wd
rm(list=ls())
set.seed(19)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#Data Folder
path <- "data_clean/"

#load in data for further subsetting
df <- read.csv(paste0(path,"ht_subset_", 1, ".csv"), stringsAsFactors=FALSE)
#by sw score and post length
df <- df[df$sw_score<.5,]
#order by match to balance
df <- df[order(-df$matched),]
n_balance <-2*(sum(df$matched))
df <- df[1:n_balance,]
df_m <- df


for(j in 2:3){
  #load in data for further subsetting
  df <- read.csv(paste0(path,"ht_subset_", j, ".csv"), stringsAsFactors=FALSE)
  #by sw score and post length
  df <- df[df$sw_score<.5,]
  #order by match to balance
  df <- df[order(-df$matched),]
  n_balance <-2*(sum(df$matched))
  df <- df[1:n_balance,]
  #append together 
  df_m <- rbind(df_m, df)
}
  #save files
write.csv(df_m, file = paste0(path,"ht_Sw_25.csv"))



# clear workspace, set seed, and set wd
rm(list=ls())
set.seed(19)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(doBy)

#Data Folder
path <- "data_clean/"
#load in datasets to create categorical data
df <- read.csv(paste0(path,"ht_Sw_25.csv"), stringsAsFactors=FALSE)
#drop ID2
df <- df[df$matched==1,]
# Collapse by post
df <- df[ !duplicated(df$ID1), ]
df <- df[,c(5:11,21)]

# Save File
write.csv(df_m, file = paste0(path,"ht_cat_Sw_25.csv"))



# #assign cores for parallel
# cl <- makeCluster(8, "SOCK") # where 3 is the number of cores used
# registerDoSNOW(cl)
# 
# df1 <- df
# df <- df1[1:100,1:16]
# 
# #do in parallel
# df$sw_score <- 0 
# df$post1_length <- 0
# df[,18] <-str_count(df[,7], "\\S+")
# options(warn=-1)
# par <- foreach(i = 1:nrow(df), .combine = rbind, .verbose = F, .errorhandling = "pass") %dopar% {
#   library(textreuse)
#   #REAL WORK
#   df[i,17] <- as.numeric(align_local(df[i,7], df[i,14], gap =0)[3])
#   #df[i,,drop=FALSE]
# }
# 
# options(warn=0)
# stopCluster(cl)
# par <- as.data.frame(par)
# df$sw_score<- par[,1]
# df$sw_score <- suppressWarnings(as.numeric(df$sw_score))
# df$sw_score <- ifelse(is.na(df$sw_score, 0, df$sw_score)

