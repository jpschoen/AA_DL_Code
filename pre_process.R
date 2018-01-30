#=============================================================#
# John Schoeneman
# Work Done For: HT Project, OSU
# Date: Summer 2017
# Work Done: create list of combinations and merge data
# Machine: MacPro OSX Yosemite
#=============================================================#


# clear workspace, set seed, and set wd
rm(list=ls())
set.seed(19)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


# load libraries

# load data
master <- read.csv("data_clean/clean_1to8.csv", stringsAsFactors=FALSE) 
master <- master[,-1]
master <- unique(master)
master$ID <- 1:length(master$Phone_Parsed) # create unique ID
master <- subset(master, master$Phone_Parsed>100000000) # only keep phone numbers with area codes
quantile(master$freq.sum, seq(0, 1, 0.01))

master <- master[master$freq.sum>5 & master$freq.sum<=617,]

write.csv(master, file = "freq_5to617.csv")

#master phone list
phone_list <- unique(master$Phone_Parsed) # unique phone numbers, should create list of numbers to parallelize


df1 <- matrix(nrow=length(phone_list),ncol=1)



phone_list_sub <- phone_list[1:length(phone_list)]
a <- 0

for(i in phone_list_sub){
  a <- a +1
  print(a) 
  #i = phone_list_sub[6]
  df <- subset(master, master$Phone_Parsed==i)#subset by phone number
  if(length(df[,1])==1){next}
  list_combo <- t(combn(df$ID,2))# create combo list if(length(df[,1])==1){next}
  df1[a,1] <- length(as.data.frame(list_combo)[,1])
} 

############ Use Smith-Waterman alignment to remove very close matches)



#Create base to bind to
i=phone_list[1]
df <- subset(master, master$Phone_Parsed==i)#subset by phone number
list_combo <- t(combn(df$ID,2))# create combo list
df1 <- as.data.frame(list_combo)
names(df1) <- c("ID1", "ID2")
df1 <- merge(df1, df, by.x = c("ID1"), by.y = c("ID"))
colnames(df1)[3:8] <- c("ID1_phone", "ID1_date", "ID1_city", "ID1_state", "ID1_post", "ID1_freq")
df1 <- merge(df1, df, by.x = c("ID2"), by.y = c("ID"))
colnames(df1)[9:14] <- c("ID2_phone", "ID2_date", "ID2_city", "ID2_state", "ID2_post", "ID2_freq")
df1$check_post1 <- apply(df1, 1, function(d) adist(d[7], d[13])/nchar(d[7]))# check of post are same
df1 <- subset(df1, df1$check_post1>.05) #check if post is more

# Apply Smith Waterman alogrithm score 
# as percentage of post one
df1$sw_score <- 0
df1$post1_length <- 0
options(warn=-1)
for(i in 1:length(df1$sw_score)){
  df1$post1_length[i] <-str_count(df1$ID1_post[1], "\\S+")
  a <- tryCatch(
    align_local(df1$ID1_post[i], df1$ID2_post[i], gap =0)[3],
    error=function(e) e)
  if(inherits(a, "error")) next
  
  #REAL WORK
  df1$sw_score[i] <- as.numeric(align_local(df1$ID1_post[i], df1$ID2_post[i], gap =0)[3])/df1$post1_length[i]
  
}
options(warn=0)

df1 <- subset(df1, df1$sw_score<1.5 ) #check if post is more
df1$matches_n <- length(df1[,1]) # how many matches are there per number
df1$matched <- 1 # create dummy that there is a match
df_m <- df1#need to bind together at end.

a <- 1
phone_list_sub <- phone_list[2:length(phone_list)]
start.time <- Sys.time()
for(i in phone_list_sub){
  
  df <- subset(master, master$Phone_Parsed==i)#subset by phone number
  print(length(df[,1]))
  if(length(df[,1])==1){next}
  list_combo <- t(combn(df$ID,2))# create combo list
  df1 <- as.data.frame(list_combo)
  names(df1) <- c("ID1", "ID2")
  a <- a+1
  print(a)
  df1 <- merge(df1, df, by.x = c("ID1"), by.y = c("ID"))
  colnames(df1)[3:8] <- c("ID1_phone", "ID1_date", "ID1_city", "ID1_state", "ID1_post", "ID1_freq")
  df1 <- merge(df1, df, by.x = c("ID2"), by.y = c("ID"))
  colnames(df1)[9:14] <- c("ID2_phone", "ID2_date", "ID2_city", "ID2_state", "ID2_post", "ID2_freq")
  df1$check_post1 <- apply(df1, 1, function(d) adist(d[7], d[13])/nchar(d[7]))# check of post are same
  df1 <- subset(df1, df1$check_post1>.5) #check if post is more
  if(length(df1[,1])==0){next}
  df1$matches_n <- length(df1[,1]) # how many matches are there per number
  df1$matched <- 1 # create dummy that there is a match
  df_m <- rbind(df_m, df1)#need to bind together at end
  
}
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken




