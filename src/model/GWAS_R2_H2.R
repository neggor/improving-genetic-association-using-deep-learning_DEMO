
# Get hidden dimension / phenotypes  --> Heritability score by dim/trait
#                                    --> R^2 per dim/trait
#                                    --> GWAS per dim/trait

# All phenotype representations follow:
# CSV with Genotype as first column.

###############################################################################
###############################################################################

# Load libraries
# library(heritability) # Hack the function to have upper and lower separated
repeatability <- function (data.vector, geno.vector, line.repeatability = FALSE, 
          covariates.frame = data.frame()) 
{
  stopifnot(length(geno.vector) == length(data.vector))
  her.frame <- data.frame(dat = data.vector, geno = geno.vector)
  her.frame <- her.frame[!is.na(her.frame$dat), ]
  number.of.covariates <- 0
  if (nrow(covariates.frame) > 0) {
    stopifnot(nrow(covariates.frame) == length(data.vector))
    cov.names <- names(covariates.frame)
    number.of.covariates <- length(cov.names)
    covariates.frame <- as.data.frame(covariates.frame[!is.na(data.vector), 
    ])
    names(covariates.frame) <- cov.names
    her.frame <- cbind(her.frame, covariates.frame)
    her.frame <- her.frame[apply(covariates.frame, 1, function(x) {
      sum(is.na(x))
    }) == 0, ]
  }
  her.frame$geno <- factor(her.frame$geno)
  n.rep.vector <- as.integer(table(her.frame$geno))
  n.geno <- length(n.rep.vector)
  average.number.of.replicates <- (sum(n.rep.vector) - sum(n.rep.vector^2)/sum(n.rep.vector))/(length(n.rep.vector) - 
                                                                                                 1)
  if (max(n.rep.vector) == 1) {
    return(list(repeatability = NA, gen.variance = NA, res.variance = NA))
  }
  else {
    if (nrow(covariates.frame) > 0) {
      av <- anova(lm(as.formula(paste("dat~", paste(cov.names, 
                                                    collapse = "+"), "+geno")), data = her.frame))
    }
    else {
      av <- anova(lm(dat ~ geno, data = her.frame))
    }
    gen.variance <- (av[[3]][1 + number.of.covariates] - 
                       av[[3]][2 + number.of.covariates])/average.number.of.replicates
    if (gen.variance < 0) {
      gen.variance <- 0
    }
    if (line.repeatability) {
      res.variance <- av[[3]][2 + number.of.covariates]/average.number.of.replicates
    }
    else {
      res.variance <- av[[3]][2 + number.of.covariates]
    }
    if (!line.repeatability) {
      F.ratio <- av[[3]][1 + number.of.covariates]/av[[3]][2 + 
                                                             number.of.covariates]
      df.1 <- n.geno - 1
      df.2 <- av[[1]][2 + number.of.covariates]
      F.L <- qf(0.025, df1 = df.1, df2 = df.2, lower.tail = TRUE)
      F.U <- qf(0.975, df1 = df.1, df2 = df.2, lower.tail = TRUE)
      conf.int.left <- (F.ratio/F.U - 1)/(F.ratio/F.U + 
                                            average.number.of.replicates - 1)
      conf.int.right <- (F.ratio/F.L - 1)/(F.ratio/F.L + 
                                             average.number.of.replicates - 1)
      if (conf.int.left < 0) {
        conf.int.left <- 0
      }
      if (conf.int.right > 1) {
        conf.int.right <- 1
      }
      if (conf.int.left > 1) {
        conf.int.left <- 1
      }
      if (conf.int.right < 0) {
        conf.int.right <- 0
      }
    }
    else {
      conf.int.left <- NA
      conf.int.right <- NA
    }
    return(list(repeatability = gen.variance/(gen.variance + 
                                                res.variance), gen.variance = gen.variance, res.variance = res.variance, 
                line.repeatability = line.repeatability, average.number.of.replicates = average.number.of.replicates, 
                lower = conf.int.left, upper = conf.int.right))
  }
}


library(argparser)
library(stringr)
library(tidyverse)
library(statgenGWAS)
library(lme4)





# FUNCTIONS;  
# rep_path : Representation path
# hd_name: Name of representation

#### BROAD SENSE HERITABILITY:
my_heritability <- function(rep_path, hd_name){
  # Gets representation -> Stores heritability by dimension
  # Parse data
  #HR <- tail(read.csv(rep_path), 331) for test set in contrastive
  HR <- read.csv(rep_path)
  geno_vector <- HR[, 1]
  
  # Remove NA
  HR <- HR[, c(apply(HR, 2, function(x) !any(is.na(x))))]
  nb_col <- dim(HR)[2]
  
  her_list <- vector(mode = 'list', nb_col - 1)
  # Loop over the phenotype dimensions
  for (pheno in 2:nb_col){
    if (pheno %% 10 == 0){
      print(pheno)
    }
    data_vector <- HR[, pheno]
    her_list[[pheno - 1]] = repeatability(data.vector = data_vector, geno.vector = geno_vector,
                  line.repeatability = F)
  }
  
  her_list = do.call(rbind, her_list)
  rownames(her_list) <- colnames(HR)[2:nb_col]
  # Return CSV with results
  write.csv(her_list, str_interp('results/${hd_name}_repeatability.csv'))
  return(as.data.frame(her_list))
}

### NARROW SENSE HERITABILITY:
my_marker_h2_means <- function(rep_path, hd_name){
  # Generate kinship matrix:
  load('data/raw/raw_data/R_data/LFN350acc_000_new_gene_annotation_ibd_kinship.RData')
  genotype_codes <- read.csv('data/raw/raw_data/genotype-codes.csv')
  km <- kinship(t(GWAS.obj$markers), method = "astle")
  
  # Load data:
  HR <- read.csv(rep_path)
  #HR <- tail(read.csv(rep_path), 331)
  # Calculate genotipic means:
  HR <- HR %>%
    group_by(Genotype) %>%
    summarize(across(.cols = everything(), .fns = mean))
  
  # Change names genotype:
  HR$Genotype <- genotype_codes[match(
    as.numeric(HR$Genotype), genotype_codes$ID.ethogenomics), 2]
  
  geno_vector <- HR[, 1]
  
  
  # Remove NA
  HR <- HR[, c(apply(HR, 2, function(x) !any(is.na(x))))]
  nb_col <- dim(HR)[2]
  
  # Placeholder
  her_matrix <- matrix(0, nb_col - 1, 3)
  colnames(her_matrix) <- c('h2', 'ci. Low', 'ci. High')
  
  # Compute:
  for (pheno in 2:nb_col){
    if (pheno %% 10 == 0){
      print(pheno)
    }
    data_vector <- pull(HR[, pheno])
    #her_list[[pheno - 1]] = repeatability(data.vector = data_vector, geno.vector = geno_vector,
    #                                      line.repeatability = F)
    
    result <- marker_h2_means(data_vector, geno_vector$Genotype, km)
    print(result)
    her_matrix[pheno - 1, ] <- c(result$h2, result$conf.int1[1], result$conf.int1[2])
   
    
  }
  return(as.data.frame(her_matrix))
  
}

### HERITABILITY FOR PCA
my_PCA_heritability <- function(rep_path, n_comp, hd_name){
  # Parse data
  HR <- read.csv(rep_path)
  geno_vector <- HR[, 1]
  nb_col <- dim(HR)[2]
  
  her_list <- vector(mode = 'list', nb_col - 1)
  
  # Remove NA
  HR <- HR[, c(apply(HR, 2, function(x) !any(is.na(x))))]
  
  PCA_transform = princomp(HR[, 2:nb_col])
  
  PCA_transform = PCA_transform$scores[, 1:n_comp]
  
  # Loop over the phenotype dimensions
  for (pheno in 1:n_comp){
    data_vector <- PCA_transform[, pheno]
    her_list[[pheno]] = repeatability(data.vector = data_vector, geno.vector = geno_vector,
                                      line.repeatability = T)
  }
  
  her_list = do.call(rbind, her_list)
  rownames(her_list) <- colnames(PCA_transform)[1:n_comp]
  # Return CSV with results
  
  write.csv(her_list, str_interp('results/${hd_name}_PCA_repeatability.csv'))
  return(her_list)
}

### VARIANCE EXPLAINED PER SNP:
my_variance_explained <- function(rep_path,
                                  hd_name,
                                  markers_path ='data/raw/raw_data/R_data/LFN350acc_000_new_gene_annotation_ibd_kinship.RData',
                                  SNP_pos = 86001){
  HR <- read.csv(rep_path)
  HR <- HR[, c(apply(HR, 2, function(x) !any(is.na(x))))]
  genotype_codes <- read.csv('data/raw/raw_data/genotype-codes.csv')
  
  load(markers_path) # Generates 'GWAS.obj'

  HR$Genotype <- # Match genotype names with markers matrix colnames.
    genotype_codes[
      match(as.numeric(HR$Genotype),
            genotype_codes$ID.ethogenomics), 2]
  
  
  # Indexing to match both arrays
  index <- match(HR$Genotype, colnames(GWAS.obj$markers[SNP_pos, ]))
  index <- index[!is.na(index)]
  
  HR <- HR[HR$Genotype %in% colnames(GWAS.obj$markers[SNP_pos, ]), ] # Just the available genotypes
  
  X <- as.factor(t(GWAS.obj$markers[SNP_pos, index])) # Get SNP value in order
  
  expl_variance <- numeric(ncol(HR) - 1)
  
  for( i in 2:ncol(HR)){
    expl_variance[i-1] <- summary(lm(HR[, i] ~ X))$r.squared
  }
  expl_variance <- as.data.frame(expl_variance)
  row_names <- colnames(HR[, 2:ncol(HR)])
  col_name <- 'R^2'
  
  rownames(expl_variance) <- row_names
  colnames(expl_variance) <- col_name
  write.csv(expl_variance, str_interp('results/${hd_name}_R2_${SNP_pos}.csv'))
  return(expl_variance)
  
}

### VARIANCE EXPLAINED PER SNP, PCA:
my_PCA_explained <- function(rep_path,
                             hd_name,
                             markers_path ='data/raw/raw_data/R_data/LFN350acc_000_new_gene_annotation_ibd_kinship.RData',
                             SNP_pos = 86001){
  
  HR <- read.csv(rep_path)

  # Generate PCA
  PCA_transform = princomp(HR[, 2:nb_col])
  
  PCA_transform = PCA_transform$scores[, 1]
  
  genotype_codes <- read.csv('data/raw/raw_data/genotype-codes.csv')
  
  HR$Genotype <- # Match genotype names with markers matrix colnames.
    genotype_codes[
      match(as.numeric(HR$Genotype),
            genotype_codes$ID.ethogenomics), 2]
  
  # Parse genetic data
  
  load(markers_path) # Generates 'GWAS.obj'
  
  Y <- t(GWAS.obj$markers[SNP_pos, match(HR$Genotype, colnames(GWAS.obj$markers[SNP_pos, ]))]) # Get SNP value in order
  
  # Calculate R^2
  expl_variance <- summary(lm(Y ~ PCA_transform))$r.squared
  
  return(expl_variance)
  
}
####################
#### GWAS STUFF:
###################

### GET ARITHMETIC AND PREDICTED MEANS (MIXED MODEL) 
get_means <- function(rep_path, hd_name, metainfo_path, storage_path){
  #########
  #Process input representation into simple arithmetic mean and
  # shrinked mean with a LMM.
  # IN -> Representation per observation, name, metainfo (plant & trial)
  # OUT -> 2 files with the two different means by genotype.
  ######
  
  ## Add trial and plant to data:
  HR <- read.csv(rep_path)
  HR$Genotype <- NULL # Cuz genotype also in metainfo
  metainfo <- read.csv(metainfo_path)
  Full_data <- cbind(metainfo, HR)
  Full_data$Trial <- as.factor(sapply(strsplit(Full_data$Trial, " "),
                                      function(x){(tail(x, 1))})) # Just remove 
                                                                  # the 'Trial '
  Full_data$Plant <- as.factor(Full_data$Plant)
  Full_data$Genotype <- as.factor(Full_data$Genotype)
  # Here the structure is Trial-Arena-Genotype-Plant-...

  # Placeholders for the means.
  predicted_means <- matrix(ncol = ncol(Full_data) - 3,
                            nrow = length(unique(Full_data$Genotype)))
  
  #print(dim(predicted_means))
  #print(colnames(Full_data)[c(3, 5:ncol(Full_data))])
  colnames(predicted_means) <- colnames(Full_data)[c(3, 5:ncol(Full_data))]
  
  empirical_means <- matrix(ncol = ncol(Full_data) - 3,
                            nrow = length(unique(Full_data$Genotype)))
  
  colnames(empirical_means) <- colnames(Full_data)[c(3, 5:ncol(Full_data))]
  
  
  i <- 0
  for (t in colnames(Full_data)[5:ncol(Full_data)]){
    print(t)
    ## Arithmetic means:
    e_means <- as.data.frame(Full_data %>% group_by(Genotype) %>% summarise('Emp.mean' = mean(.data[[t]])))
    
    if (i == 0){
      empirical_means[, 'Genotype'] <- as.numeric(levels(e_means$Genotype))
    }
    
    empirical_means[, t] <- e_means$Emp.mean
    ##


    ## Run mixed model:
    my_model <- lmer(get(t) ~ Genotype + (1|Plant) + (1|Trial), data = Full_data)
    p_means <- as.data.frame(lsmeans::lsmeans(my_model, ~ 1 + Genotype))
    if (i == 0){
      predicted_means[, 'Genotype'] <- as.numeric(levels(p_means$Genotype))
    }
   
    predicted_means[, t] <- p_means$lsmean
    ##
    
    i <- i + 1
  }
  #stop()
  
  ## Predicted means
  write.csv(as.data.frame(predicted_means),
            str_interp('results/${storage_path}/Predicted_means_${hd_name}.csv'),
            row.names = FALSE)
  
  ## Arithmetic means:
  write.csv(as.data.frame(empirical_means),
            str_interp('results/${storage_path}/Empirical_means_${hd_name}.csv'),
            row.names = FALSE)
            
}

### PERFORM GWAS IN ALL DIMENSIONS
my_GWAS <- function(means_path,
                    hd_name,
                    storage_path,
                    high_resolution = FALSE){
  
  # High resolution uses 1M SNP array. Memory problems do not allow to clean
  # the array so the result is problematic.
  
  if (!high_resolution){
    
  markers_path ='data/raw/raw_data/R_data/LFN350acc_000_new_gene_annotation_ibd_kinship.RData'
  } else{
    markers_path ='data/raw/raw_data/R_data/hapmap_imputed_snps_1M.RData'
  }
  
  # Load mean traits
  traits <- read.csv(means_path)
  
  # Load 
  genotype_codes <- read.csv('data/raw/raw_data/genotype-codes.csv')
  load(markers_path) # Generates 'GWAS.obj' or gData1M
  
  # Remove NA & transform to factor
  traits$Genotype <- as.numeric(traits$Genotype)
  traits <- traits[complete.cases(traits$Genotype),]
  clean_traits <- traits[, c(apply(traits, 2, function(x) !any(is.na(x))))]
  
  gc()
  
  # Prepare mapping
  if (!high_resolution){
    my_map <- GWAS.obj$map[, c(1,2)]
    colnames(my_map) <- c("chr", "pos")
    rownames(my_map) <- colnames(t(GWAS.obj$markers))
    my_GData <- createGData(geno = t(GWAS.obj$markers), map = my_map)
    km <- kinship(t(GWAS.obj$markers), method = "astle")
  
    clean_traits$Genotype <- 
      genotype_codes[
        match(as.numeric(clean_traits$Genotype),
              genotype_codes$ID.ethogenomics), 2]
    
    colnames(clean_traits)[1] <- 'genotype' 
    
    my_GData <- createGData(gData = my_GData, pheno = clean_traits)
    
    my_GData_clean <- codeMarkers(my_GData,
                                  impute = FALSE,
                                  verbose = TRUE,
                                  nMissGeno = 0.01,
                                  nMiss = 0.01)
    
  } else{
    my_map <- gData1M$map[, c(1,2)]
    colnames(my_map) <- c("chr", "pos")
    my_GData <-createGData(geno = gData1M$markers, map = my_map)
    km <- kinship(gData1M$markers, method = "astle")
    

    clean_traits$Genotype <- 
      genotype_codes$Magnus.nordborg.ID[
        match(as.numeric(clean_traits$Genotype),
              genotype_codes$Magnus.nordborg.ID)]
    
    colnames(clean_traits)[1] <- 'genotype' 
    
    # not enough memory to "clean" it in this case...
    my_GData_clean <- createGData(gData = my_GData, pheno = clean_traits)
    
  }
  
  
  
 
  

  for (my_trait in colnames(traits)[2:ncol(traits)]){
    
    print(str_interp('Running single trait GWAS for ${my_trait}...'))
      
    GWAS_1 <- runSingleTraitGwas(   gData = my_GData_clean,
                                    kin = km,
                                    thrType = "small",
                                    nSnpLOD = 10,
                                    useMAF = FALSE,
                                    MAC = 10, traits = c(my_trait))
                                    
    
    png(filename= str_interp('results/${storage_path}/${hd_name}_${my_trait}_GWAS.png'))
    plot(GWAS_1)
    dev.off()
    
    results <- GWAS_1$GWAResult$clean_traits[
      order(GWAS_1$GWAResult$clean_traits$pValue), ]
    
    write.csv(results, str_interp('results/${storage_path}/${hd_name}_${my_trait}_GWAS.csv'))
    
  }
  return(results)
}
 

###############################################################################
###############################################################################
# RUN EXPERIMENTS:

## Handcrafted:
x <- 'data/processed/Hidden_representations/Hand_features/handcrafted_dimensions.csv'

my_heritability(x,
                'results/Hand_features/Handcrafted_dimensions')
my_variance_explained(x,
                      'results/Hand_features/Handcrafted_dimensions')

get_means(rep_path = x,
          hd_name = 'Handcrafted_dimensions',
          metainfo_path = 'data/processed/Hidden_representations/Hand_features/handcrafted_dimensions_meta_info.csv',
          storage_path= 'Hand_features')

my_GWAS(means_path = 'results/Self-Hand_features/Predicted_means_Handcrafted_dimensions.csv',
        hd_name = 'Handcrafted_dimensions',
        storage_path = 'Hand_features')

## Supervised:
x <- 'data/processed/Hidden_representations/supervised/IT_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_snp86001_BS20.csv'

my_heritability(x,
                'results/Supervised/IT_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_snp86001_BS20')

my_variance_explained(x,
                      'results/Supervised/IT_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_snp86001_BS20')

get_means(rep_path = x,
          hd_name = 'IT_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_snp86001_BS20',
          metainfo_path = 'data/processed/Hidden_representations/supervised/IT_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_snp86001_BS20_metainfo.csv',
          storage_path= 'Supervised')

my_GWAS(means_path = 'results/Supervised/Predicted_means_IT_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_snp86001_BS20.csv',
        hd_name = 'IT_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_snp86001_BS20',
        storage_path = 'Supervised')

## Self-Supervised:

x = 'data/processed/Hidden_representations/self-supervised/MINIMAL_VAE_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_BS32.csv'

my_heritability(x,
                'results/Self-Supervised/MINIMAL_VAE_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_BS32')

my_variance_explained(x,
                      'results/Self-Supervised/MINIMAL_VAE_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_BS32')

get_means(metainfo_path =  'data/processed/Hidden_representations/self-supervised/MINIMAL_VAE_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_BS32_metainfo.csv',
          rep_path = 'data/processed/Hidden_representations/self-supervised/MINIMAL_VAE_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_BS32.csv',
          hd_name = 'MINIMAL_VAE_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_BS32',
          storage_path = 'Self-Supervised')

my_GWAS('results/Self-Supervised/Predicted_means_MINIMAL_VAE_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_BS32.csv',
        'predicted_MINIMAL_VAE_C0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_GS1_BS32',
        storage_path = 'Self-Supervised')

## Contrastive:


x = 'data/processed/Hidden_representations/Contrastive/MINIMAL_Contrastive_C1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_BS128.csv'

my_heritability(x,
                'results/Contrastive/MINIMAL_Contrastive_C1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_BS128')
my_variance_explained(x,
                      'results/Contrastive/MINIMAL_Contrastive_C1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_BS128')


get_means(metainfo_path =  'data/processed/Hidden_representations/Contrastive/MINIMAL_Contrastive_C1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_BS128_metainfo.csv',
          rep_path = x,
          hd_name = 'MINIMAL_Contrastive_C1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_BS128',
          storage_path = 'Contrastive')

my_GWAS('results/Contrastive/Predicted_means_MINIMAL_Contrastive_C1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_BS128.csv',
        'MINIMAL_Contrastive_C1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20_D1_BS128',
        storage_path = 'Contrastive')


