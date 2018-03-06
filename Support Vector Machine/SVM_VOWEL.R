

runVowelSVM = function(costExp, gammaExp){
  
  model <- svm(Class~., data= train, kernel = "radial", gamma=2^gammaExp, cost= 2^costExp)
  
  
  prediction <- predict(model, test[,-13])
  
  confusionMatrix <- table(pred = prediction, true = test$Class)
  
  agreement <- prediction == test$Class
  a <- table(agreement)
  
  accuracy <- prop.table(table(agreement))
  
  print(accuracy)
  # print(unname(a[2]) / unname(a[1]))
  return(unname(a[2]) / (unname(a[1]) + unname(a[2])))
}

library (e1071)

vowels <- read.csv("vowel.csv", head=TRUE, sep=",")

costExp = seq(from=-1, to=5, by=2)
gammaExp = seq(from=-3, to=3, by=2)


highestAccuracy = 0
highestACostExp = 0
highestAGammaEXP = 0

rows <- 1:nrow(vowels)

testRows <- sample(rows, trunc(length(rows) * 0.3))

test <-vowels[testRows,]
train <- vowels[-testRows, ]

for(i in costExp){
  for(j in gammaExp){
    cat(sprintf("cost: 2^%i\n gamma: 2^%i\n", i, j))
    accuracy = runVowelSVM(i, j)
    print(accuracy)
    if(accuracy > highestAccuracy){
      highestAccuracy = accuracy
      highestACostExp = i
      highestAGammaEXP = j
    }
  }
}
cat(sprintf("highest accuracy: %f\n cost: 2^%i\n gamma: 2^%i\n", highestAccuracy, highestACostExp, highestAGammaEXP))
