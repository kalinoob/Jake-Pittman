
# Load necessary libraries
library(shiny)
library(xgboost)
library(dplyr)

# Load the trained model
xgb_model <- readRDS("~/Documents/xgb_tuned.rds")

# Load the saved feature names
feature_names <- readRDS("~/Documents/feature_names.rds")

# Function to predict spend
predict_spend_for_clicks <- function(job_title, city, state_code, desired_clicks, days_live = 1) {
  
  # Create a new dataframe for this prediction
  new_data <- data.frame(
    title = job_title,
    city = city,
    state_code = state_code,
    days_live = days_live,
    applies = 0,
    spend = 0,
    Duration = days_live,
    stringsAsFactors = FALSE
  )
  
  # Segment assignment based on job title
  new_data$Segment <- ifelse(grepl("Registered Nurse|RN|Essential", new_data$title, ignore.case = TRUE), "Essential RNs",
                             ifelse(grepl("Therapist|Therapy|Specialist", new_data$title, ignore.case = TRUE), "Therapy Specialists",
                                    ifelse(grepl("Support|Nursing Assistant|CNA", new_data$title, ignore.case = TRUE), "Nursing & Therapy Support",
                                           ifelse(grepl("Manager|Director|Lead|Leadership", new_data$title, ignore.case = TRUE), "Leadership and management",
                                                  "Admin & Misc."))))
  
  # Derived Features
  new_data$Clicks_Per_Day <- desired_clicks / (days_live + 1)
  
  # Manual Encoding for Categorical Variables
  # City Encoding
  known_cities <- c("Phoenix", "Chicago", "New York") # Add all known cities from training here
  new_data$city_encoded <- ifelse(new_data$city %in% known_cities, new_data$city, "Unknown_City")
  
  # State Encoding
  known_states <- c("AZ", "NY", "CA") # Add all known state codes from training here
  new_data$state_code_encoded <- ifelse(new_data$state_code %in% known_states, new_data$state_code, "Unknown_State")
  
  # Segment Encoding
  known_segments <- c("Essential RNs", "Therapy Specialists", "Nursing & Therapy Support", "Leadership and management", "Admin & Misc.")
  new_data$Segment_encoded <- ifelse(new_data$Segment %in% known_segments, new_data$Segment, "Unknown_Segment")
  
  # Create dummy variables manually
  new_data <- new_data %>%
    mutate(
      city_Phoenix = ifelse(city_encoded == "Phoenix", 1, 0),
      city_Chicago = ifelse(city_encoded == "Chicago", 1, 0),
      city_NewYork = ifelse(city_encoded == "New York", 1, 0),
      city_Unknown = ifelse(city_encoded == "Unknown_City", 1, 0),
      
      state_AZ = ifelse(state_code_encoded == "AZ", 1, 0),
      state_NY = ifelse(state_code_encoded == "NY", 1, 0),
      state_CA = ifelse(state_code_encoded == "CA", 1, 0),
      state_Unknown = ifelse(state_code_encoded == "Unknown_State", 1, 0),
      
      segment_EssentialRNs = ifelse(Segment_encoded == "Essential RNs", 1, 0),
      segment_TherapySpecialists = ifelse(Segment_encoded == "Therapy Specialists", 1, 0),
      segment_NursingSupport = ifelse(Segment_encoded == "Nursing & Therapy Support", 1, 0),
      segment_Leadership = ifelse(Segment_encoded == "Leadership and management", 1, 0),
      segment_AdminMisc = ifelse(Segment_encoded == "Admin & Misc.", 1, 0),
      segment_Unknown = ifelse(Segment_encoded == "Unknown_Segment", 1, 0)
    )
  
  # Remove original categorical columns to avoid confusion
  new_data <- new_data %>%
    select(-title, -city, -state_code, -Segment, -city_encoded, -state_code_encoded, -Segment_encoded)
  
  # Add missing columns that were present during training but may be missing in new_data
  missing_cols <- setdiff(feature_names, colnames(new_data))
  for (col in missing_cols) {
    new_data[[col]] <- 0
  }
  
  # Ensure columns are in the same order as the training set
  new_data <- new_data[, feature_names, drop = FALSE]
  
  # Convert to matrix
  new_data_matrix <- as.matrix(new_data)
  
  # Make prediction
  log_predicted_spend <- predict(xgb_model, newdata = new_data_matrix)
  predicted_spend <- exp(log_predicted_spend) - 1
  
  return(predicted_spend)
}

# Define the UI
ui <- fluidPage(
  titlePanel("Spend Prediction for Job Listings"),
  
  sidebarLayout(
    sidebarPanel(
      textInput("job_title", "Job Title:", value = "Registered Nurse Hospice"),
      textInput("city", "City:", value = "Summerville"),
      textInput("state_code", "State Code:", value = "SC"),
      numericInput("desired_clicks", "Desired Clicks:", value = 35, min = 1),
      actionButton("predict_btn", "Predict Spend")
    ),
    
    mainPanel(
      textOutput("prediction_output")
    )
  )
)

# Define the server logic
server <- function(input, output) {
  observeEvent(input$predict_btn, {
    job_title <- input$job_title
    city <- input$city
    state_code <- input$state_code
    desired_clicks <- input$desired_clicks
    
    # Predict spend based on user inputs
    predicted_spend <- predict_spend_for_clicks(job_title, city, state_code, desired_clicks)
    
    # Render the prediction result
    output$prediction_output <- renderText({
      paste("The estimated budget to achieve", desired_clicks, "clicks is: $", round(predicted_spend *10, 2))
    })
  })
}

# Run the application
shinyApp(ui = ui, server = server)





