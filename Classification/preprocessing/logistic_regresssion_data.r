df = read.csv("../train_ISIC_2020.csv", header = TRUE)

## general
logitMod <- glm(target ~ sex + age_approx + anatom_site_general_challenge, data=df, family=binomial(link="logit"))
summary(logitMod)

## age
logitMod2 <- glm(target ~ age_approx, data=df, family=binomial(link="logit"))
summary(logitMod2)

## sex (no effect)
logitMod3 <- glm(target ~ sex, data=df, family=binomial(link="logit"))
summary(logitMod3)

## age+sex
logitMod4 <- glm(target ~ sex+age_approx, data=df, family=binomial(link="logit"))
summary(logitMod4)
##  ln(p/(1-p)) = -6.607236 + 0.307200*(if male) + 0.045152*age 
## ⇒ p = σ(+6.607236 - 0.307200*(if male) - 0.045152*age)
## ⇒ p = 1/(1+exp(+6.607236 - 0.307200*(if male) - 0.045152*age))




 
