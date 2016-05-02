# R script set up and loading in data -------------------------------------

# Set working directory
setwd("~/Documents/R/Expedia Hotel Recommendations")

# Load in useful libraries
library(data.table)
library(dplyr)
library(caret)
library(ggplot2)
library(GGally)

# Load in training and test data
train <- fread('input/train.csv', header=TRUE)
test <- fread('input/test.csv', header=TRUE)
sample <- fread('input/sample_submission.csv', header=TRUE)


# Training data too large - Split train to be more manageable for data analysis
set.seed(2016)
splitIndex <- createDataPartition(train$hotel_cluster, p = .02,
                                  list = FALSE,
                                  times = 1)

expediaTrain <- train[splitIndex[, 1], ]



# Data cleaning -----------------------------------------------------------

# # Count number of NAs in all training data
na_count <- sapply(expediaTraining, function(y) sum(length(which(is.na(y)))))
na_count # Seems like only the orig_destination_distance variable contains NAs
na_count[7] / length(train$hotel_cluster) # 36% of observations have NAs as distances - not trivial

# I commented out some lines of code which I do not use in the main analysis.

# # Let's take a closer look at the distance variable
# hist.distance <- qplot(expediaTrain$orig_destination_distance, geom = "histogram")
# hist.distance # 3 distinct chunks - can split into 3 different categories for classification purposes if need be
# hist.distance <- ggplot_build(hist.distance)
# hist.distance$data # Cut-off values roughly seem to be 196.579 and 2948.685

# First let's generate a distance variable where we replace the NAs with the average distance
dist1 <- ifelse(is.na(expediaTrain$orig_destination_distance),
                mean(expediaTrain$orig_destination_distance, na.rm = TRUE),
                expediaTrain$orig_destination_distance)

# # Let's generate a categorical distance variable where we cut it into 5 categories
# dist2 <- cut(expediaTrain$orig_destination_distance, 5)
# dist2 <- ifelse(is.na(dist2), 99, dist2)
# 
# # Let's generate another categorical distance variable where it is 1 if below 200, 2 if between 200 and 3000, 3 if over 3000 and 99 if NA
# dist3 <- ifelse(expediaTrain$orig_destination_distance < 200, 1,
#                ifelse((expediaTrain$orig_destination_distance >= 200) & (expediaTrain$orig_destination_distance <= 3000), 2,
#                ifelse(expediaTrain$orig_destination_distance > 3000, 3, 99)))
# dist3[is.na(dist3)] <- 99

# Let's append this back to the training data
expediaTrain$dist1 <- dist1
# expediaTrain$dist2 <- dist2
# expediaTrain$dist3 <- dist3

# Remove full train data to save computer memory
rm(train)

# Rename expediaTrain to train
train <- expediaTrain
rm(expediaTrain)

# Converting target dependent variable into a factor
train$hotel_cluster <- as.factor(train$hotel_cluster)
levels(train$hotel_cluster) <- make.names(levels(factor(train$hotel_cluster)))



# Feature engineering -----------------------------------------------------

# Generate month of holiday
train$checkInMonth <- as.Date(train$srch_ci)
train$checkInMonth <- month(train$checkInMonth)
train$checkInMonth[is.na(train$checkInMonth)] <- as.integer(names(sort(table(train$checkInMonth), decreasing=TRUE)[1]))

# Generate measure to indicate popularity of hotel_cluster in each destination
hotel.popularity <- aggregate(train$is_booking, by = list(train$srch_destination_id, train$hotel_cluster), length)
temp <- aggregate(hotel.popularity$x, by = list(hotel.popularity$Group.1), sum)
hotel.popularity <- merge(hotel.popularity, temp, by = "Group.1")
rm(temp)
hotel.popularity$most.popular <- round(hotel.popularity$x.x / hotel.popularity$x.y, digits = 2)
hotel.popularity <- hotel.popularity[, c(1, 2, 3, 5)]
names(hotel.popularity) <- c("srch_destination_id", "hotel_cluster", "hotel.popularity.count", "hotel.popularity.prop")

# Generate tag to indicate top 5 hotel_clusters in each destination
temp <- hotel.popularity %>%
           group_by(srch_destination_id) %>%
           top_n(n = 5, wt = hotel.popularity.count)

temp$top5.dummy <- 1
temp <- temp[, c(1, 2, 5)]
hotel.popularity <- merge(hotel.popularity, temp, 
                           by = c("srch_destination_id", "hotel_cluster"),
                           all.x = TRUE)
hotel.popularity$top5.dummy[is.na(hotel.popularity$top5.dummy)] <- 0

train <- merge(train, hotel.popularity, by = c("srch_destination_id", "hotel_cluster"))



# Data analysis and visualisation -----------------------------------------

# Plotting the distribution of hotel clusters in training data
count <- table(train$hotel_cluster)
barplot(count)

# Count number of times each hotel_cluster is booked in the training data - to see most popular hotel
most.bookings <- aggregate(train$is_booking, by = list(train$hotel_cluster), sum)
barplot(most.bookings$x, names.arg = most.bookings$Group.1)

# Looking at most popular hotel_cluster by looking at % of times that it was booked
most.bookings.prop <- aggregate(train$is_booking, by = list(train$hotel_cluster), 
                                FUN = function(x) {sum(x)/length(x)})
barplot(most.bookings.prop$x, names.arg = most.bookings.prop$Group.1)

# A common idea on Kaggle forums was to look at the most popular hotel by destination. Let's see if we can generate the number of times a hotel_cluster has appeared per destination
most.popular <- aggregate(train$is_booking, 
                          by = list(train$srch_destination_id, train$hotel_cluster), length)
most.popular <- most.popular[order(most.popular$Group.1, most.popular$x), ]

# Let's see if we can infer anything as well from the proportions of most.popular
temp <- aggregate(most.popular$x, by = list(most.popular$Group.1), sum)
most.popular.prop <- merge(most.popular, temp, by = "Group.1")
most.popular.prop$most.popular <- round(most.popular.prop$x.x / most.popular.prop$x.y, digits = 2)



# Fitting a model ---------------------------------------------------------

# Further split training dataset into 2 for cross-validation purposes
set.seed(2016)
trainIndex <- createDataPartition(train$hotel_cluster, p = .8,
                                  list = FALSE,
                                  times = 1)

expediaTraining <- train[trainIndex[, 1], ]
expediaTesting <- train[-trainIndex[, 1], ]

# Fitting an initial random forest model on all features
library(randomForest)

fit1 <- randomForest(hotel_cluster ~ user_location_country + hotel_market 
                    + srch_destination_id	+ dist1 + checkInMonth, 
                    data = expediaTraining,
                    na.action = na.omit)
save(fit, file = "model1.rda") # Save for easy loading later

# Commented out fit2 because no time to figure out how to infer hotel.popularity.count and top5.dummy
# for test dataset
# fit2 <- randomForest(hotel_cluster ~ user_location_country + hotel_market 
#                     + srch_destination_id	+ dist1 + checkInMonth
#                     + hotel.popularity.count + top5.dummy, 
#                     data = expediaTraining,
#                     na.action = na.omit)
# save(fit, file = "model2.rda") # Save for easy loading later



# Cross validation --------------------------------------------------------

# Prediction using test set
pred <- predict(fit1, expediaTesting, type = "prob")

# Formatting into submission
submit <- apply(pred, 1, FUN = function(x) {
  i1 <- x!=0
  i2 <- order(-x[i1])
  head(colnames(pred)[i1][i2], 5)})

submit <- do.call(rbind, submit)
submit <- submit[, 1:5]
submit <- replace(submit, is.na(submit), "")
submit <- as.data.frame(submit)

# Running a MAP@5 function on our predictions...but first format required inputs
predicted <- as.data.frame(sapply(submit, gsub,pattern = "X", replacement = ""))
predicted <- as.data.frame(sapply(submit, as.numeric))
predicted <- split(predicted, seq(nrow(predicted)))
names(predicted) <- NULL

actuals <- expediaTesting$hotel_cluster
actuals <- as.data.frame(sapply(actuals, gsub, pattern = "X", replacement = ""))
actuals <- as.data.frame(sapply(actuals, as.numeric))
actuals <- split(actuals, seq(nrow(actuals)))
names(actuals) <- NULL

library(Metrics)
mapk(5, actuals, predicted) # MAP@5 = 0.25 for fit1 and 0.64 for fit2

# However, the above assumes that we have data on hotel_cluster, whilst we actually don't.
# Ideally, the way that I would go about the anlaysis if I have more time is to predict the 
# hotel.popularity.count and top5.dummy using features that I hae access to via either OLS or
# logit models. As a workaround for the timebeing, I will assume that top5.dummy = 1 for all test
# observations, and hotel.popularity.count = average by hotel.popularity.count for the time being.

temp <- aggregate(hotel.popularity$hotel.popularity.count, 
                  by = list(hotel.popularity$srch_destination_id), max)

names(temp) <- c("srch_destination_id", "hotel.popularity.count")

expediaTesting2 <- expediaTesting[, .SD, .SDcols = c(1:26)]
expediaTesting2 <- merge(expediaTesting2, temp, by = "srch_destination_id")
expediaTesting2$top5.dummy <- 1

# Prediction using new test set
pred <- predict(fit, expediaTesting2, type = "prob")

# Formatting into submission
submit <- apply(pred, 1, FUN = function(x) {
  i1 <- x!=0
  i2 <- order(-x[i1])
  head(colnames(pred)[i1][i2], 5)})

submit <- do.call(rbind, submit)
submit <- submit[, 1:5]
submit <- replace(submit, is.na(submit), "")
submit <- as.data.frame(submit)

# Running a MAP@5 function on our predictions...but first format required inputs
predicted <- as.data.frame(sapply(submit, gsub,pattern = "X", replacement = ""))
predicted <- as.data.frame(sapply(submit, as.numeric))
predicted <- split(predicted, seq(nrow(predicted)))
names(predicted) <- NULL

actuals <- expediaTesting$hotel_cluster
actuals <- as.data.frame(sapply(actuals, gsub, pattern = "X", replacement = ""))
actuals <- as.data.frame(sapply(actuals, as.numeric))
actuals <- split(actuals, seq(nrow(actuals)))
names(actuals) <- NULL

library(Metrics)
mapk(5, actuals, predicted) # MAP@5 = 0.12


# Prepping test dataset for prediction ------------------------------------

# Generating distance variable for test dataset
dist1.test <- ifelse(is.na(test$orig_destination_distance), 
                     mean(test$orig_destination_distance, na.rm = TRUE),
                     test$orig_destination_distance)

test$dist1 <- dist1.test

# Generating check in month variable for test dataset
test$checkInMonth <- as.Date(test$srch_ci)
test$checkInMonth <- month(test$checkInMonth)
test$checkInMonth[is.na(test$checkInMonth)] <- as.integer(names(sort(table(test$checkInMonth), decreasing=TRUE)[1]))

# Generate measure to indicate popularity of hotel_cluster in each destination
hotel.popularity.test <- aggregate(test$is_booking, by = list(test$srch_destination_id, test$hotel_cluster), length)

temp <- aggregate(hotel.popularity.test$x, by = list(hotel.popularity$Group.1), sum)

hotel.popularity.test <- merge(hotel.popularity.test, temp, by = "Group.1")
rm(temp)

hotel.popularity.test$most.popular <- round(hotel.popularity.test$x.x / hotel.popularity.test$x.y, digits = 2)

hotel.popularity.test <- hotel.popularity.test[, c(1, 2, 3, 5)]

names(hotel.popularity.test) <- c("srch_destination_id", "hotel_cluster", "hotel.popularity.count", "hotel.popularity.prop")

# Generate tag to indicate top 5 hotel_clusters in each destination
temp <- hotel.popularity.test %>%
           group_by(srch_destination_id) %>%
           top_n(n = 5, wt = hotel.popularity.count)

temp$top5.dummy <- 1
temp <- temp[, c(1, 2, 5)]
hotel.popularity.test <- merge(hotel.popularity.test, temp, 
                          by = c("srch_destination_id", "hotel_cluster"),
                          all.x = TRUE)

hotel.popularity.test$top5.dummy[is.na(hotel.popularity.test$top5.dummy)] <- 0

test <- merge(test, hotel.popularity.test, by = c("srch_destination_id", "hotel_cluster"))



# Generating submission file ----------------------------------------------

# Prediction using test set
pred.test <- predict(fit1, test, type = "prob")

# Formatting into submission
submit <- apply(pred.test, 1, FUN = function(x) {
  i1 <- x!=0
  i2 <- order(-x[i1])
  head(colnames(pred)[i1][i2], 5)})

submit <- do.call(rbind, submit)
submit <- submit[, 1:5]
submit <- replace(submit, is.na(submit), "")
submit <- as.data.frame(submit)

hotel_cluster <- paste(submit$V1, 
                       submit$V2, 
                       submit$V3, 
                       submit$V4, 
                       submit$V5, sep = " ")

hotel_cluster <- gsub("X", "", hotel_cluster)
id <- test$id

# Generating submission file
submission <- cbind(id, hotel_cluster)
submission <- as.data.frame(submission)
write.csv(submission, file = "submission.csv", row.names = FALSE)
