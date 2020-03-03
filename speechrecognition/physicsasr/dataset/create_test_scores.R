library(haven)

args = commandArgs(trailingOnly=TRUE)

if (length(args) != 2){
  stop("You must supply paths to the input file and output file", call.=FALSE)
}

data1 <- read_spss(args[1])

# Select finns
data1 <- data1[data1$LAND == "F",]

# Calculate average pre and post test score per teacher
test_scores <- do.call("rbind", lapply(split(data1, data1$SCHULE), function(teacher){
  averages <- sapply(split(teacher, teacher$Zeit), function(x){
    mean(rowSums(x[,-c(1:7, 62:150)], na.rm = TRUE))
  })
  data.frame("teacher" = teacher$SCHULE[1], "pre" = averages[1], "post" = averages[2])
}))

write.csv(test_scores, file = args[2], row.names = FALSE)