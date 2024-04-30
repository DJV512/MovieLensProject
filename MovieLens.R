# Our final goal
goal <- 0.8649

##########################################################
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Set seed for repeatability
set.seed(1, sample.kind="Rounding")

# Split up edx dataset into training and testing sets
test_index <- createDataPartition(edx$rating, times=1, p=0.1, list=FALSE)
edx_train <- edx[-test_index,]
edx_test_temp <- edx[test_index,]

# Ensure that there are no movies in the testing set that aren't in the training set
edx_test <- edx_test_temp %>%
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")
removed <- anti_join(edx_test_temp, edx_test)
edx_train <- rbind(edx_train, removed)

# Remove unneeded variables from memory
rm(edx_test_temp, removed, test_index)

# Defining our error function that will determine model accuracy
RMSE <- function(ratings, y_hat) {
  sqrt(mean((ratings-y_hat)^2))
}

# Calculate the average rating across the entire edcx_train data set
average_rating <- mean(edx_train$rating)
average_rating

# Graph the distribution of movie ratings in the edx_train data set
edx_train %>% ggplot(aes(rating)) + geom_histogram(color="black", fill="blue", bins=10) + xlab("Movie Rating") + ylab("Total")

# If we predict that every movie in the test set has the average rating, how far off are we?
RMSE_1 <- RMSE(edx_test$rating, average_rating)
RMSE_1

# Update results table
results_table <- tibble(Model = c("1. Average Rating","", "Goal"), RMSE = c(RMSE_1, "", goal)) %>% knitr::kable()
results_table

# Determine user bias in isolation
user_bias <- edx_train %>% group_by(userId) %>% summarize(u_bias = mean(rating-average_rating))

# Graph user bias
user_bias %>% ggplot(aes(u_bias)) + geom_histogram(color="black", fill="blue", bins=10) + xlab("User bias") + ylab("Total")

# Determine movie bias in isolation
movie_bias <- edx_train %>% group_by(movieId) %>% summarize(m_bias = mean(rating-average_rating))

# Graph movie bias
movie_bias %>% ggplot(aes(m_bias)) + geom_histogram(color="black", fill="blue", bins=10) + xlab("Movie bias") + ylab("Total")

# Adding movie effects to average rating
y_hat2 <- edx_test %>% left_join(movie_bias, by = "movieId") %>% mutate(pred=(average_rating + m_bias)) %>% pull(pred)

RMSE_2 <- RMSE(edx_test$rating, y_hat2)
RMSE_2

# Update results table
results_table <- tibble(Model = c("1. Average Rating", "2. Movie Effects Added", "", "Goal"), RMSE = c(RMSE_1, RMSE_2, "", goal)) %>% knitr::kable()
results_table

# Recalculating user bias after accounting for movie bias
user_bias <- edx_train %>% left_join(movie_bias, by="movieId") %>% group_by(userId) %>% summarize(u_bias = mean(rating-average_rating-m_bias))

# Accounting for movie and user bias, how does our model do?
y_hat3 <- edx_test %>% left_join(movie_bias, by = "movieId") %>% left_join(user_bias, by = "userId") %>% mutate(pred=(average_rating + m_bias + u_bias)) %>% pull(pred)

RMSE_3 <- RMSE(edx_test$rating, y_hat3)
RMSE_3

# Update results table
results_table <- tibble(Model = c("1. Average Rating", "2. Movie Effects Added", "3. Movie and User Effects Added", "", "Goal"), RMSE = c(RMSE_1, RMSE_2, RMSE_3, "", goal)) %>% knitr::kable()
results_table

# Updating the data sets to add a column for release year
edx_train <- edx_train %>% mutate(age = 2024 - as.numeric(str_sub(title, start=-5, end=-2)))

edx_test <- edx_test %>% mutate(age = 2024 - as.numeric(str_sub(title, start=-5, end=-2)))

final_holdout_test <- final_holdout_test %>% mutate(age = 2024 - as.numeric(str_sub(title, start=-5, end=-2)))

# Confirming that data mutation was successful
head(edx_train)

# Calculating age-related metrics in the edx_train data set
ratings_per_age <- edx_train %>% group_by(age) %>% summarize(ratings_count = n(), average_rating = mean(rating))

# Graphing number of ratings for movies of a certain age
ratings_per_age %>% ggplot(aes(age, ratings_count)) + geom_point() + xlab("Movie age") + ylab("Number of ratings")

# Graphing average rating of movies based on age
ratings_per_age %>% ggplot(aes(age, average_rating)) + geom_point() + geom_smooth() + xlab("Movie age") + ylab("Average rating")

# If we also account for age bias, on top of movie and user biases, how does our model do?
age_bias <- edx_train %>% left_join(movie_bias, by="movieId") %>% left_join(user_bias, by="userId") %>% group_by(age) %>% summarize(a_bias = mean(rating-average_rating-m_bias-u_bias))

y_hat4 <- edx_test %>% left_join(movie_bias, by = "movieId") %>% left_join(user_bias, by = "userId") %>% left_join(age_bias, by="age") %>% mutate(pred=(average_rating + m_bias + u_bias + a_bias)) %>% pull(pred)

RMSE_4 <- RMSE(edx_test$rating, y_hat4)
RMSE_4

# Update results table
results_table <- tibble(Model = c("1. Average Rating", "2. Movie Effects Added", "3. Movie and User Effects Added", "4. Movie, User, and Age Effects Added", "", "Goal"), RMSE = c(RMSE_1, RMSE_2, RMSE_3, RMSE_4, "", goal)) %>% knitr::kable()
results_table

# Calculate genre bias in isolation
genre_bias <- edx_train %>% group_by(genres) %>% summarize(g_bias = mean(rating-average_rating))

# Graph genre bias
genre_bias %>% ggplot(aes(g_bias)) + geom_histogram(color="black", fill="blue", bins=10) + xlab("Genre bias") + ylab("Total")

# Recalculate genre bias after accounting for previously determined biases
genre_bias <- edx_train %>% left_join(movie_bias, by="movieId") %>% left_join(user_bias, by="userId") %>% left_join(age_bias, by="age") %>% group_by(genres) %>% summarize(g_bias = mean(rating-average_rating-m_bias-u_bias-a_bias))

# How does adding genre effects affect the accuracy of the movie?
y_hat5 <- edx_test %>% left_join(movie_bias, by = "movieId") %>% left_join(user_bias, by = "userId") %>% left_join(age_bias, by="age") %>% left_join(genre_bias, by="genres") %>% mutate(pred=(average_rating + m_bias + u_bias + a_bias + g_bias)) %>% pull(pred)

RMSE_5 <- RMSE(edx_test$rating, y_hat5)
RMSE_5

# Update results table
results_table <- tibble(Model = c("1. Average Rating", "2. Movie Effects Added", "3. Movie and User Effects Added", "4. Movie, User, and Age Effects Added", "5. Movie, User, Age, and Genre Effects Added", "", "Goal"), RMSE = c(RMSE_1, RMSE_2, RMSE_3, RMSE_4, RMSE_5, "", goal)) %>% knitr::kable()
results_table

# Using regularization to determine the lambda value that minimizes RMSE in our model
lambdas <- seq(2, 8, 0.25)

regularization <- function(x) {
  
  movie_bias <- edx_train %>% group_by(movieId) %>% summarize(m_bias = sum(rating-average_rating)/(n()+x))
  
  user_bias <- edx_train %>% left_join(movie_bias, by="movieId") %>% group_by(userId) %>% summarize(u_bias = sum(rating-average_rating-m_bias)/(n()+x))
  
  age_bias <- edx_train %>% left_join(movie_bias, by="movieId") %>% left_join(user_bias, by="userId") %>% group_by(age) %>% summarize(a_bias = sum(rating-average_rating-m_bias-u_bias)/(n()+x))
  
  genre_bias <- edx_train %>% left_join(movie_bias, by="movieId") %>% left_join(user_bias, by="userId") %>% left_join(age_bias, by="age") %>% group_by(genres) %>% summarize(g_bias = sum(rating-average_rating-m_bias-u_bias-a_bias)/(n()+x))
  
  y_hat <- edx_test %>% left_join(movie_bias, by="movieId") %>% left_join(user_bias, by="userId") %>% left_join(age_bias, by="age") %>% left_join(genre_bias, by="genres") %>% mutate(pred=(average_rating + m_bias + u_bias + a_bias + g_bias)) %>% pull(pred)
  
  RMSE(edx_test$rating, y_hat)
  
}

RMSEs_regular <- sapply(lambdas, regularization)

plot(lambdas, RMSEs_regular)

# Using the best value of lambda, how does our regularized model do?
best_lambda <- lambdas[which.min(RMSEs_regular)]
best_lambda

RMSE_6 <- RMSEs_regular[best_lambda]
RMSE_6

# Update results table
results_table <- tibble(Model = c("1. Average Rating", "2. Movie Effects Added", "3. Movie and User Effects Added", "4. Movie, User, and Age Effects Added", "5. Movie, User, Age, and Genre Effects Added", "6. Adding Regularization", "", "Goal"), RMSE = c(RMSE_1, RMSE_2, RMSE_3, RMSE_4, RMSE_5, RMSE_6, "", goal)) %>% knitr::kable()
results_table


# Using our final model on the final_holdout_test data set
movie_bias <- edx_train %>% group_by(movieId) %>% summarize(m_bias = sum(rating-average_rating)/(n()+best_lambda))

user_bias <- edx_train %>% left_join(movie_bias, by="movieId") %>% group_by(userId) %>% summarize(u_bias = sum(rating-average_rating-m_bias)/(n()+best_lambda))

age_bias <- edx_train %>% left_join(movie_bias, by="movieId") %>% left_join(user_bias, by="userId") %>% group_by(age) %>% summarize(a_bias = sum(rating-average_rating-m_bias-u_bias)/(n()+best_lambda))

genre_bias <- edx_train %>% left_join(movie_bias, by="movieId") %>% left_join(user_bias, by="userId") %>% left_join(age_bias, by="age") %>% group_by(genres) %>% summarize(g_bias = sum(rating-average_rating-m_bias-u_bias-a_bias)/(n()+best_lambda))

y_hat_final <- final_holdout_test %>% left_join(user_bias, by = "userId") %>% left_join(movie_bias, by = "movieId") %>% left_join(age_bias, by="age") %>% left_join(genre_bias, by="genres") %>% mutate(pred=(average_rating + m_bias + u_bias + a_bias + g_bias)) %>% pull(pred)

# Final model RMSE on final_holdout_test data set
RMSE_final <- RMSE(final_holdout_test$rating, y_hat_final)
RMSE_final
