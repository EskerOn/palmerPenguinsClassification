install.packages("palmerpenguins")
library(palmerpenguins)
df <- data.frame(penguins)

write.csv(df,"C:\\Users\\alexe\\Desktop\\penguins.csv", row.names = FALSE)