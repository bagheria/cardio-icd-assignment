library(xlsx)
library(dplyr)
library(tidyverse)
library(stringr)

data_fr <- read.csv("Data/icd_for_codes.csv")

# I48
data_I48 <- data_fr %>%
  group_by(ID, text) %>%
  summarize(samenvatting = paste(samenvatting, collapse = 'THISISSPACE'),
            age_norm     = unique(age_norm),
            gender       = unique(gender),
            I48 = paste(I48, collapse = ','))

data_I48$I48 <- sapply(strsplit(data_I48$I48, ","),
                       function(x) paste(unique(x), collapse = ","))
data_I48$I48 <- ifelse(str_detect(data_I48$I48, ","), 1, data_I48$I48)

table(data_I48$I48)
path = "D:/Github/ICD10 Classification/"
write_csv(data_I48, paste(path, "I48.csv", sep = ""))


# I10
data_I10 <- data_fr %>%
  group_by(ID, text) %>%
  summarize(samenvatting = paste(samenvatting, collapse = 'THISISSPACE'),
            age_norm     = unique(age_norm),
            gender       = unique(gender),
            I10 = paste(I10, collapse = ','))

data_I10$I10 <- sapply(strsplit(data_I10$I10, ","),
                       function(x) paste(unique(x), collapse = ","))
data_I10$I10 <- ifelse(str_detect(data_I10$I10, ","), 1, data_I10$I10)

table(data_I10$I10)
path = "D:/Github/ICD10 Classification/"
write_csv(data_I10, paste(path, "I10.csv", sep = ""))


# I21
data_I21 <- data_fr %>%
  group_by(ID, text) %>%
  summarize(samenvatting = paste(samenvatting, collapse = 'THISISSPACE'),
            age_norm     = unique(age_norm),
            gender       = unique(gender),
            I21 = paste(I21, collapse = ','))

data_I21$I21 <- sapply(strsplit(data_I21$I21, ","),
                       function(x) paste(unique(x), collapse = ","))
data_I21$I21 <- ifelse(str_detect(data_I21$I21, ","), 1, data_I21$I21)

table(data_I21$I21)
path = "D:/Github/ICD10 Classification/"
write_csv(data_I21, paste(path, "I21.csv", sep = ""))


# I25
data_I25 <- data_fr %>%
  group_by(ID, text) %>%
  summarize(samenvatting = paste(samenvatting, collapse = 'THISISSPACE'),
            age_norm     = unique(age_norm),
            gender       = unique(gender),
            I25 = paste(I25, collapse = ','))

data_I25$I25 <- sapply(strsplit(data_I25$I25, ","),
                       function(x) paste(unique(x), collapse = ","))
data_I25$I25 <- ifelse(str_detect(data_I25$I25, ","), 1, data_I25$I25)

table(data_I25$I25)
path = "D:/Github/ICD10 Classification/"
write_csv(data_I25, paste(path, "I25.csv", sep = ""))


# I42
data_I42 <- data_fr %>%
  group_by(ID, text) %>%
  summarize(samenvatting = paste(samenvatting, collapse = 'THISISSPACE'),
            age_norm     = unique(age_norm),
            gender       = unique(gender),
            I42 = paste(I42, collapse = ','))

data_I42$I42 <- sapply(strsplit(data_I42$I42, ","),
                       function(x) paste(unique(x), collapse = ","))
data_I42$I42 <- ifelse(str_detect(data_I42$I42, ","), 1, data_I42$I42)

table(data_I42$I42)
path = "D:/Github/ICD10 Classification/"
write_csv(data_I42, paste(path, "I42.csv", sep = ""))


# E11
data_E11 <- data_fr %>%
  group_by(ID, text) %>%
  summarize(samenvatting = paste(samenvatting, collapse = 'THISISSPACE'),
            age_norm     = unique(age_norm),
            gender       = unique(gender),
            E11 = paste(E11, collapse = ','))

data_E11$E11 <- sapply(strsplit(data_E11$E11, ","),
                       function(x) paste(unique(x), collapse = ","))
data_E11$E11 <- ifelse(str_detect(data_E11$E11, ","), 1, data_E11$E11)

table(data_E11$E11)
path = "D:/Github/ICD10 Classification/"
write_csv(data_E11, paste(path, "E11.csv", sep = ""))


# E78
data_E78 <- data_fr %>%
  group_by(ID, text) %>%
  summarize(samenvatting = paste(samenvatting, collapse = 'THISISSPACE'),
            age_norm     = unique(age_norm),
            gender       = unique(gender),
            E78 = paste(E78, collapse = ','))

data_E78$E78 <- sapply(strsplit(data_E78$E78, ","),
                       function(x) paste(unique(x), collapse = ","))
data_E78$E78 <- ifelse(str_detect(data_E78$E78, ","), 1, data_E78$E78)

table(data_E78$E78)
path = "D:/Github/ICD10 Classification/"
write_csv(data_E78, paste(path, "E78.csv", sep = ""))


# N18
data_N18 <- data_fr %>%
  group_by(ID, text) %>%
  summarize(samenvatting = paste(samenvatting, collapse = 'THISISSPACE'),
            age_norm     = unique(age_norm),
            gender       = unique(gender),
            N18 = paste(N18, collapse = ','))

data_N18$N18 <- sapply(strsplit(data_N18$N18, ","),
                       function(x) paste(unique(x), collapse = ","))
data_N18$N18 <- ifelse(str_detect(data_N18$N18, ","), 1, data_N18$N18)

table(data_N18$N18)
path = "D:/Github/ICD10 Classification/"
write_csv(data_N18, paste(path, "N18.csv", sep = ""))


# I50
data_I50 <- data_fr %>%
  group_by(ID, text) %>%
  summarize(samenvatting = paste(samenvatting, collapse = 'THISISSPACE'),
            age_norm     = unique(age_norm),
            gender       = unique(gender),
            I50 = paste(I50, collapse = ','))

data_I50$I50 <- sapply(strsplit(data_I50$I50, ","),
                       function(x) paste(unique(x), collapse = ","))
data_I50$I50 <- ifelse(str_detect(data_I50$I50, ","), 1, data_I50$I50)

table(data_I50$I50)
path = "D:/Github/ICD10 Classification/"
write_csv(data_I50, paste(path, "I50.csv", sep = ""))


# Z95
data_Z95 <- data_fr %>%
  group_by(ID, text) %>%
  summarize(samenvatting = paste(samenvatting, collapse = 'THISISSPACE'),
            age_norm     = unique(age_norm),
            gender       = unique(gender),
            Z95 = paste(Z95, collapse = ','))

data_Z95$Z95 <- sapply(strsplit(data_Z95$Z95, ","),
                       function(x) paste(unique(x), collapse = ","))
data_Z95$Z95 <- ifelse(str_detect(data_Z95$Z95, ","), 1, data_Z95$Z95)

table(data_Z95$Z95)
path = "D:/Github/ICD10 Classification/"
write_csv(data_Z95, paste(path, "Z95.csv", sep = ""))
