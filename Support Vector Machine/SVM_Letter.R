runLetterSVM = function(costExp, gammaExp){
  
  rows <- 1:nrow(letters)
  
  testRows <- sample(rows, trunc(length(rows) * 0.3))
  
  test <-letters[16001:20000,]
  train <- letters[1:16000, ]
  
  model <- svm(letter~., data= train, kernel = "radial", gamma=2^gammaExp, cost= 2^costExp)
  
  
  prediction <- predict(model, test[,-1])
  
  confusionMatrix <- table(pred = prediction, true = test$letter)
  
  agreement <- prediction == test$letter
  a <- table(agreement)
  
  accuracy <- prop.table(table(agreement))
  
  print(accuracy)
  # print(unname(a[2]) / unname(a[1]))
  return(unname(a[2]) / (unname(a[1]) + unname(a[2])))
}

library (e1071)

letters <- read.csv("letters.csv", head=TRUE, sep=",")

costExp = seq(from=-1, to=5, by=2)
gammaExp = seq(from=-3, to=3, by=2)

# costExp = -1
# gammaExp = -3

highestAccuracy = 0
highestACostExp = 0
highestAGammaEXP = 0

for(i in costExp){
  for(j in gammaExp){
    cat(sprintf("cost: 2^%i\n gamma: 2^%i\n", i, j))
    accuracy = runLetterSVM(i, j)
    print(accuracy)
    if(accuracy > highestAccuracy){
      highestAccuracy = accuracy
      highestACostExp = i
      highestAGammaEXP = j
    }
  }
}
cat(sprintf("highest accuracy: %f\n cost: 2^%i\n gamma: 2^%i\n", highestAccuracy, highestACostExp, highestAGammaEXP))


