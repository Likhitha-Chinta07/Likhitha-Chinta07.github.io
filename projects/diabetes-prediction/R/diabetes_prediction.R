# Modernized reproduction of the original academic R workflow
# Original project: Diabetes Prediction, Ch.Lakshmi Likhitha

# 1. Load the CSV using read.csv, replacing the old file.choose() approach.
dia_prediction <- read.csv("data/diabetes.csv", stringsAsFactors = FALSE)

# 2. Validate the expected structure.
required_columns <- c(
  "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
  "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
)
stopifnot(all(required_columns %in% names(dia_prediction)))

# 3. Preserve the original project's 500 / 268 split for historical comparison.
training_data <- dia_prediction[1:500, ]
testing_data <- dia_prediction[501:768, ]

# 4. Keep the original Decision Tree idea, but use a maintained package.
# install.packages("rpart")
library(rpart)

tree_model <- rpart(
  factor(Outcome) ~ ., 
  data = training_data,
  method = "class",
  control = rpart.control(cp = 0.01)
)

# 5. Generate predictions and calculate the historical-style accuracy.
predictions <- predict(tree_model, testing_data[, required_columns[1:8]], type = "class")
accuracy <- mean(predictions == factor(testing_data$Outcome))

print(tree_model)
print(confusion.table <- table(Predicted = predictions, Actual = testing_data$Outcome))
print(sprintf("Decision Tree accuracy: %.4f", accuracy))

# Note: the original report recorded 79.47761% using the older tree package.
# This script is a cleaner reproduction and should not be assumed to reproduce
# the historical result exactly until it is executed against the same data.
