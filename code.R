##--- References ====
# The dataset is an edited version of the following: https://www.kaggle.com/datasets/rounakbanik/pokemon/data
# adding missing values to dataset (height and weight of alolan forms): https://bulbapedia.bulbagarden.net/wiki/Main_Page

##--- Setup ====
pokemonUnedited <- read.csv("https://raw.githubusercontent.com/ScoDix/Pokemon/main/pokemon.csv")

library(dplyr)
library(randomForest)
library(rpart)
library(caret)
library(ggplot2)
library(GGally)
library(rpart)
library(ggparty)
library(DescTools)

##--- Data preparation ====
# Converting categorical variables into factors:
pokemon <- pokemonUnedited %>%
  mutate_if(is.character, ~ ifelse(. == "", "None", as.character(.))) %>% 
  mutate_at(vars(type1, type2, ability_1, ability_2, ability_hidden, percentage_male), factor, ordered = FALSE)

# Colours for each type
normal <- '#A8A77A'
fire <- '#EE8130'
water <- '#6390F0'
electric <- '#F7D02C'
grass <- '#7AC74C'
ice <- '#96D9D6'
fighting <- '#C22E28'
poison <- '#A33EA1'
ground <- '#E2BF65'
flying <- '#A98FF3'
psychic <- '#F95587'
bug <- '#A6B91A'
rock <- '#B6A136'
ghost <- '#735797'
dragon <- '#6F35FC'
dark <- '#705746'
steel <- '#B7B7CE'
fairy <- '#D685AD'

##--- Code for Figure 1 of report ====
# Graph of base stat totals for dragon, bug, and electric types.
# This graph only has one axis, so I'm shifting all the data points by some random distance perpendicular to the axis to make
# all the points visible. The 152 is the number of data points. It can be verified by running the following commented line:
# #nrow(pokemon %>% filter(type1 == "Dragon" | type1 == "Bug" | type1 == "Electric"))
noise_for_1D_graph <- data.frame(noise = rnorm(152, mean = 0, sd = 0.1))

ggplot(pokemon %>% filter(type1 == "Dragon" | type1 == "Bug" | type1 == "Electric") %>% mutate(noise = noise_for_1D_graph$noise), aes(y=noise,x=base_total))+
  geom_point(aes(colour=factor(type1)))+
  ylab("")+
  xlab("Base stat total")+
  labs(colour="")+
  scale_color_manual(values=c(bug, dragon, electric),labels=c("Bug","Dragon", "Electric"))+
  theme_minimal()+
  theme(panel.grid.major.y = element_blank())+ #, legend.position = "bottom"
  scale_y_continuous(breaks=NULL)

##--- Explanatory: variables to remove at the outset ====

# All of the type effectiveness variables: the set of type effectivenesses is unique for every
# combination of types. It just doesn't tell you which way round the types are. In other words,
# given any valid set of type effectivenesses, you (literally, it doesn't take long for a human)
# can work out the type combination.
# name: this is unique for every pokemon, Interestingly, one can infer a pokemon's main type
# from the etymology of the name. E.g. squirtle = squirt + turtle --> water or electrode --> electric
# training a model to work out a pokemon's type based on its name is beyond the scope of this report
# despite simple examples for humans such as squirtle and electrode.
# Japanese name: same reason.

# the variables I am most interested in are the base stats; this is because certain types
# are associated with certain stats:
# examples: steel types have high defence, as do rock
# flying types have high speed
# fighting types have high attack
# psychic and dragon types have high special attack
# dragon types have high base stat totals

##--- Separating into training and testing ====
# First, remove the variables that are unique identifiers of Pokemon.
pokemonNames <- pokemon %>% 
  select(pokedex_number, name, classification)
pokemonData <- pokemon %>% 
  select(-pokedex_number, -name, -classification)

set.seed(42)
indices_of_train = sample(1:862, 690) # 80/20 split of training to testing data
train = pokemonData[indices_of_train,]
test = pokemonData[-indices_of_train,]

##--- not used in final analysis: function to aggregate abilities by type ====
# This function replaces all the abilities with a type, based on what the most common type is
# for pokemon with that ability: e.g. "Overgrow" is unique to grass type pokemon, so this is
# replaced by "grass". The motivation for this is to significantly reduce the number of factors
# in each ability variable.

aggregate_ability_by_type <- function(df) {
  df <- df %>%
    group_by(ability_1) %>%
    mutate(ability_type_1 = as.factor(Mode(type1)[1])) %>%
    ungroup() %>% 
    group_by(ability_2) %>% 
    mutate(ability_type_2 = as.factor(Mode(type1)[1])) %>% 
    ungroup() %>% 
    group_by(ability_hidden) %>% 
    mutate(ability_type_hidden = as.factor(Mode(type1)[1])) %>% 
    ungroup() %>% 
    select(-ability_1, -ability_2, -ability_hidden)
  return(df)
}

#train = aggregate_ability_by_type(pokemon)
#test = aggregate_ability_by_type(test)

##--- Full tree + a pruned tree for figure 2 of report ====
fitControl = trainControl(method = "cv", number = 10)

# Decision tree with all variables
set.seed(42)
fullTree = train(type1 ~ .,# - ability_1 - ability_2 - ability_hidden,
                             data = train,
                             method = "rpart",
                             trControl = fitControl,
                             tuneLength = 20) # test 20 values of the complexity parameter
print(fullTree)
autoplot(as.party(fullTree$finalModel))

# A pruned tree for figure 2 of report, purely for display purposes
prunedfullTree = train(type1 ~ .,# - ability_1 - ability_2 - ability_hidden,
                 data = train,
                 method = "rpart",
                 trControl = fitControl,
                 tuneGrid = expand.grid(cp = 0.0125)) # this value of cp was found to cut the tree to 12 levels
print(prunedfullTree)
autoplot(as.party(prunedfullTree$finalModel))

##--- Testing full tree ====
# Testing the predictions of "fullTree"
predictions <- predict(fullTree,
                       newdata = test,
                       type = "raw")
test$pred_type1 <- predictions # add the predictions to the test data

# If the prediction of type is equal to the actual type return 1 (true), else return 0 (false).
# The accuracy is the mean of all the 0s and 1s.
accuracy <- mean(predictions == test$type1)
print(accuracy)

# add the variables "pokedex_number", "name" and "classification" back to train and test
train <- bind_cols(train, pokemonNames[indices_of_train,])
test <- bind_cols(test, pokemonNames[-indices_of_train,])
# reorder the columns
train <- train %>% select(pokedex_number, name, generation, is_legendary, classification, everything())
test <- test %>% select(pokedex_number, name, generation, is_legendary, classification, type1, pred_type1, everything())

# calculate accuracy for each type
# i.e. what percent of grass type pokemon were predicted as being grass type?
fullAccuracyType <- test %>%
  group_by(type1) %>%
  summarise(count = n(),
            matches = sum(type1 == pred_type1)) %>%
  mutate(accuracy = matches / count)

##--- No Abilities tree ====
# Reset train and test
train = pokemonData[indices_of_train,]
test = pokemonData[-indices_of_train,]

# Decision tree without abilities
set.seed(42)
noAbTree = train(type1 ~ . - ability_1 - ability_2 - ability_hidden,
                 data = train,
                 method = "rpart",
                 trControl = fitControl,
                 tuneLength = 20) # test 20 values of the complexity parameter
print(noAbTree)
autoplot(as.party(noAbTree$finalModel))

##--- Testing no abilities tree ====
noAbPredictions <- predict(noAbTree,
                       newdata = test,
                       type = "raw")
test$noab_pred_type1 <- noAbPredictions

# Calculate no abilities tree accuracy in the same way as with the full tree
noab_accuracy <- mean(test$noab_pred_type1 == test$type1)
print(noab_accuracy)

# add the variables "pokedex_number", "name" and "classification" back to train and test
train <- bind_cols(train, pokemonNames[indices_of_train,])
test <- bind_cols(test, pokemonNames[-indices_of_train,])
# reorder the columns
train <- train %>% select(pokedex_number, name, generation, is_legendary, classification, everything())
test <- test %>% select(pokedex_number, name, generation, is_legendary, classification, type1, noab_pred_type1, everything())

# by type accuracy:
noabAccuracyType <- test %>%
  group_by(type1) %>%
  summarise(count = n(),
            matches = sum(type1 == noab_pred_type1)) %>%
  mutate(accuracy = matches / count)

##--- Figure 3 of the report (comparing full tree and no abilities tree on their type accuracy) ====
# Create a table with all the type-accuracies of both trees
fullAccuracyType <- fullAccuracyType %>% 
  mutate(tree = "full")
noabAccuracyType <- noabAccuracyType %>% 
  mutate(tree = "noab")
accuracyType <- rbind(fullAccuracyType,noabAccuracyType)

#plot it
accuracyType$type1 <- factor(accuracyType$type1, levels = c("Dark","Normal","Bug", "Grass", "Water", "Fire", "Dragon", "Fighting", "Electric", "Ghost", "Steel", "Rock", "Ice", "Poison", "Psychic", "Fairy", "Ground"))
ggplot(accuracyType, aes(x = type1, y = 100 * accuracy, fill = tree)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(x = "Type", y = "Accuracy %", fill = "") +
  theme_minimal() +
  ylim(0,100) +
  theme(axis.text.x = element_text(angle = 35, hjust = 0.75)) +
  scale_fill_manual(values = c("#0868ac","#7bccc4"), labels = c("All variables", "No abilities"))

##--- Random Forest ====
# Reset train and test data
train = pokemonData[indices_of_train,]
test = pokemonData[-indices_of_train,]

# Random forest with 22 to 35 variables randomly selected at each node
set.seed(42)
Forest22to35 = train(type1 ~ .,
               data = train,
               method = "rf",
               trControl = fitControl,
               tuneGrid = expand.grid(mtry=seq(22,35)))
Forest <- Forest22to35
print(Forest)

##--- Testing random forest ====
ForestPredictions <- predict(Forest,
                            newdata = test,
                            type = "raw")

# Calculate accuracy in the same way
accuracy <- mean(ForestPredictions == test$type1)
print(accuracy)

# add the variables "pokedex_number", "name" and "classification" back to train
train <- bind_cols(train, pokemonNames[indices_of_train,])
# create new test data with the predictions
test_with_pred <- bind_cols(test, pokemonNames[-indices_of_train,])
test_with_pred$pred_type1 <- ForestPredictions
# reorder the columns
train <- train %>% select(pokedex_number, name, generation, is_legendary, classification, everything())
test_with_pred <- test_with_pred %>% select(pokedex_number, name, generation, is_legendary, classification, type1, pred_type1, everything())

# Calculate accuracy of random forest by type
ForestAccuracyType <- test_with_pred %>%
  group_by(type1) %>%
  summarise(count = n(),
            matches = sum(type1 == pred_type1)) %>%
  mutate(accuracy = matches / count)

##--- Figure 5 of the report ====
ForestAccuracyType$type1 <- factor(ForestAccuracyType$type1, levels = ForestAccuracyType$type1[order(-ForestAccuracyType$accuracy)])
ggplot(ForestAccuracyType, aes(x = type1, y = 100 * accuracy)) +
  geom_bar(stat = "identity", fill = "#0868ac") +
  labs(x = "Type", y = "Accuracy %") +
  theme_minimal() +
  ylim(0,100) +
  theme(axis.text.x = element_text(angle = 35, hjust = 0.75))