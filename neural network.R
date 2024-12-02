# Install packages if you haven't already
install.packages(c("readxl", "dplyr", "caret", "neuralnet"))

# Load the required libraries
library(readxl)
library(dplyr)
library(caret)
library(neuralnet)

# Load the dataset
data <- read_excel("/Users/jake.pittman/Documents/MSA_Predict_App/2024 data model.xlsx")

# Inspect the data
head(data)
str(data)   
summary(data)

# Visualize the relationship between spend and clicks
plot(data$spend, data$clicks, main = "Spend vs Clicks")

# Check the dimensions of the dataset
nrow(data)

# Feature Engineering
data <- data %>%
  mutate(
    Clicks_Per_Day = clicks / days_live,
    Spend_Per_Click = spend / clicks
  )

# Handle missing values (check for NA values)
colSums(is.na(data))

# Apply feature scaling
data$spend_scaled <- scale(data$spend)
data$clicks_scaled <- scale(data$clicks)

# Apply dummy variable encoding
dummies <- dummyVars(~ ., data = data)
data_encoded <- as.data.frame(predict(dummies, newdata = data))

# Ensure necessary columns (e.g., Clicks, Spend, etc.) exist
if (!"clicks" %in% colnames(data) | !"spend" %in% colnames(data)) {
  stop("Required columns 'clicks' or 'spend' are missing!")
}

# Define target column (Clicks per $10 of Spend)
data$Clicks_Per_10_Spend <- data$clicks / (data$spend / 10)

# Remove any rows with infinite or NaN values
data <- data[is.finite(data$Clicks_Per_10_Spend), ]

# Split the data into training (80%) and testing (20%) sets
set.seed(123)  # For reproducibility
trainIndex <- createDataPartition(data$Clicks_Per_10_Spend, p = 0.8, list = FALSE)

# Create training and testing datasets
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Check the target column distribution
summary(trainData$Clicks_Per_10_Spend)
summary(testData$Clicks_Per_10_Spend)

# Cap outliers at the 99th percentile for key features
cap_spend <- quantile(trainData$spend, 0.99)
cap_clicks <- quantile(trainData$clicks, 0.99)
cap_days_live <- quantile(trainData$days_live, 0.99)

trainData$spend <- pmin(trainData$spend, cap_spend)
trainData$clicks <- pmin(trainData$clicks, cap_clicks)
trainData$days_live <- pmin(trainData$days_live, cap_days_live)

# Normalize numerical features (clicks, spend, and days_live)
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
numerical_cols <- c("spend", "clicks", "days_live")
trainData[numerical_cols] <- lapply(trainData[numerical_cols], normalize)
testData[numerical_cols] <- lapply(testData[numerical_cols], normalize)

# Normalize the target column
max_value <- max(trainData$Clicks_Per_10_Spend)
min_value <- min(trainData$Clicks_Per_10_Spend)

trainData$Clicks_Per_10_Spend <- (trainData$Clicks_Per_10_Spend - min_value) / (max_value - min_value)
testData$Clicks_Per_10_Spend <- (testData$Clicks_Per_10_Spend - min_value) / (max_value - min_value)

# Re-run the neural network
nn <- neuralnet(
  formula = Clicks_Per_10_Spend ~ spend + clicks + days_live,
  data = trainData,
  hidden = c(4),  # Single layer with 3 neurons
  linear.output = TRUE,
  stepmax = 1e8   # Increase to 10,000,000
)


# Plot the neural network model
plot(nn)

# Make predictions on the test data
predictions <- compute(nn, testData[numerical_cols])$net.result

# Check the performance (correlation between predictions and actual values)
cor(predictions, testData$Clicks_Per_10_Spend)

# Make predictions on the test set
predictions <- compute(nn, testData[, c("spend", "clicks", "days_live")])$net.result

# Denormalize predictions if the target was normalized
predictions <- predictions * (max_value - min_value) + min_value

# Evaluate performance
mse <- mean((predictions - testData$Clicks_Per_10_Spend)^2)
rmse <- sqrt(mse)

cat("MSE:", mse, "\n")
cat("RMSE:", rmse, "\n")

# Correlation
cor(predictions, testData$Clicks_Per_10_Spend)

plot(testData$Clicks_Per_10_Spend, predictions,
     main = "Predicted vs Actual",
     xlab = "Actual Clicks Per $10 Spend",
     ylab = "Predicted Clicks Per $10 Spend")
abline(0, 1, col = "red")  # Add a perfect prediction line
