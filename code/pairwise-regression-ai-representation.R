# %%
library(tidyverse)
library(lme4)
library(dplyr)
library(ggplot2)
# install.packages("glmmTMB")
library(glmmTMB)
library(tibble)
library(emmeans)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

## Load data
model_name = 'claude-3-sonnet-v2'
all_data <- read_csv("~/Documents/PhD/ai-representation/data/demo/claude-3-sonnet-v2-pairwise_regression.csv")

all_data$age_group <- factor(all_data$age_group, 
                             levels = c("18-24", "25-34", "35-44", "45-54", "55-64", "65+"), ordered = TRUE)

all_data$income <- factor(all_data$income, 
                          levels = c('Under $30,000','$30,000-$49,999', '$50,000-$99,999', '$100,000-$199,999','Over $200,000'), 
                          ordered = TRUE)

all_data$education <- factor(all_data$education, 
                             levels = c('High school diploma or less', 'Some college, no degree', 'Associate degree', "Bachelor's degree",
                                        'Graduate or professional degree'), ordered = TRUE)

all_data$same_condition_binary <- factor(
  all_data$same_condition_binary,
  levels = c(1, 0),
  labels = c("Yes", "No")
)



contr.sum.with.na <- function(x) contr.sum(length(levels(x)))

for (col in c("political_affiliation", "gender", "race", "income", "age_group", "education")) {
  all_data[[col]] <- droplevels(factor(all_data[[col]]))
  contrasts(all_data[[col]]) <- contr.sum.with.na(all_data[[col]])
}

## Fit models and extract coefficients
results <- list()

## FOR PAIRWISE Conditions


# Fit logistic mixed-effects model
model <- glmmTMB(
  flip_binary ~ same_condition_binary * (gender + age_group + income + political_affiliation + race + education) + (1 | policy_id),
  data = all_data,
  family = binomial(link = "logit")
)

summary(model)


# 1. your demographic vars
predictors <- c(
  "gender",
  "age_group",
  "income",
  "political_affiliation",
  "race",
  "education"
)

# 2. a lookup for nice facet titles
label_map <- c(
  gender                 = "Gender",
  age_group              = "Age Group",
  income                 = "Income",
  political_affiliation  = "Political Affiliation",
  race                   = "Race",
  education              = "Education"
)

# 3. build the OR + CI + stars data.frame
or_df <- bind_rows(lapply(predictors, function(var) {
  emm  <- emmeans(model,
                  as.formula(paste0("~ same_condition_binary | ", var)),
                  type = "response")
  ctr  <- contrast(emm, method = "revpairwise")
  df   <- as.data.frame(summary(ctr, infer = TRUE))
  
  # stash var & level
  df$variable <- var
  df$level    <- df[[var]]
  
  # add significance stars
  df$sig <- with(df, case_when(
    p.value <  .001 ~ "***",
    p.value <  .01  ~ "**",
    p.value <  .05  ~ "*",
    TRUE            ~ ""
  ))
  
  df
}))

or_df_small = or_df %>% select(odds.ratio, variable, level, sig) %>% mutate(model = model_name)
write.csv(or_df_small, paste0("../data/demo/or_df_small_", model_name, ".csv"), row.names = FALSE)


# # 4. plot it
# ggplot(or_df,
#        aes(x = level,
#            y = odds.ratio,
#            ymin = asymp.LCL,
#            ymax = asymp.UCL)) +
#   geom_pointrange() +
#   # stars just above the errorbar
#   geom_text(aes(label = sig,
#                 y = asymp.UCL * 1.05),
#             vjust = 0, size = 5) +
#   facet_wrap(~ variable,
#              scales = "free_x",
#              ncol   = 2,
#              labeller = labeller(variable = label_map)) +
#   scale_y_log10(breaks = c(0.25, 0.5, 1, 2, 5),
#                 labels = c("0.25", "0.5", "1", "2", "5"),
#                 expand = expansion(mult = c(0.4, 0.5)) ) +
#   labs(
#     x     = NULL,
#     y     = "Odds Ratio (Yes vs No; 95% CI)",
#     title = "Same Condition Effect by Demographic Subgroup for Claude-3-sonnet-v2"
#   ) +
#   theme_minimal(base_size = 12) +
#   theme(
#     axis.text.x  = element_text(angle = 45, hjust = 1),
#     strip.text   = element_text(face = "bold")
#   )
# 
