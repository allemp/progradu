library(haven)
library(reshape)
data1 <- read_spss("data/raw/CKPT_1dim_mit_Anker.sav")

data2 <- read_spss("data/raw/CKPT_Komplexitat_Inhalt.sav")

data3 <- read_spss("data/raw/SQST+CKPT+KFT_final_CG7.sav")

data1 <- data1[data1$LAND == "F",]

data <- transform(data3, ID = colsplit(ID, split = "-", names = c("country","schooltype","school","class","student")))

data <- data[data$ID$country == "F",]

schools <- split(data, data$ID$school)
teachers <- do.call(rbind,lapply(schools, function(x){
  c(x$ID$school[1], mean(x$Gain, na.rm = TRUE))
}))

colnames(teachers) <- c("id", "learning_gain")

write.csv(teachers,"data/learning_gain.csv", row.names = FALSE)

student <- data1[data1$ID == 	"F-0-01-01-01",-c(1:7,112:150)]

rowSums(student, na.rm = TRUE)
