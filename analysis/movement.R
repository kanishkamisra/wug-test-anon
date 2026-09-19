# please open the .Rproj file in the home dir of 
# the repo so that the paths resolve.

library(tidyverse)
library(scales)
library(ggnewscale)
library(ggdist)
library(ggtext)
library(grid)
library(glue)
library(knitr)

# results_dir = "results/movement-analysis/Qwen_Qwen3-VL-2B-Instruct"
results_dir = "results/eval/movement-analysis/Qwen_Qwen3-VL-4B-Instruct"

real_nouns <- read_csv(glue("{results_dir}/sg_pl_reduced.csv")) %>%
  rename(number = label) %>%
  mutate(
    number = factor(number, c("sg", "pl"))
  )

real_nouns %>%
  ggplot(aes(x,y, color = number)) +
  geom_point(alpha = 0.1) +
  scale_color_manual(name = "Known Noun\nType", values = c("#a6611a", "#018571")) +
  theme_classic(base_size = 17)+
  theme(
    axis.ticks = element_blank(),
    axis.title = element_blank(),
    legend.position = "none",
    axis.text = element_blank(),
    panel.background = element_rect(fill = "transparent", colour = NA),
    plot.background  = element_rect(fill = "transparent", colour = NA)
  )

ggsave("figures/demo-sg-pl-pca.svg", height = 2.15, width = 2.39, dpi = 300)


wug_pca <- bind_rows (
  read_csv(glue("{results_dir}/wug_wugs_reduced_text.csv")) %>%
    mutate(modality='Language'),
  read_csv(glue("{results_dir}/wug_wugs_reduced_image.csv")) %>%
    mutate(modality='Vision')
)

# font_add_google("Inconsolata", "Inconsolata")
# showtext_auto()


wug_pca %>%
  pivot_wider(names_from = stage, values_from = c(x, y)) %>%
  mutate(
    type = case_when(
      type == "wug" ~ "sg",
      TRUE ~ "pl"
    ),
    type = factor(type, c("sg", "pl"))
  ) %>%
  ggplot() + 
  
  # --- 1. REAL NOUNS ---
  geom_point(data = real_nouns, aes(x, y, color = number, shape = number), alpha = 0.08) +
  scale_color_manual(
    name = "Real Noun Type", 
    values = c("#a6611a", "#018571"),
    # Define guide directly inside the scale to avoid ggnewscale errors
    guide = guide_legend(
      title.position = "top", 
      title.hjust = 0.5, 
      override.aes = list(alpha = 1)
    )
  ) +
  scale_shape_manual(
    name = "Real Noun Type", 
    values = c(16, 17),
    guide = guide_legend(
      title.position = "top", 
      title.hjust = 0.5, 
      override.aes = list(alpha = 1)
    )
  ) +
  
  # --- 2. RESET SCALES ---
  new_scale_color() +
  
  # --- 3. NOVEL NOUNS ---
  geom_segment(
    aes(
      x = x_init, y = y_init,
      xend = x_final, yend = y_final,
      color = type
    ), 
    arrow = arrow(length = unit(0.1, "cm")),
    linewidth = 0.6, alpha = 0.6
  ) +
  scale_color_manual(
    name = "Novel Noun Type", 
    values = c("#a6611a", "#018571"),
    guide = guide_legend(
      title.position = "top", 
      title.hjust = 0.5, 
      override.aes = list(alpha = 1, linewidth = 0.6)
    )
  ) +
  
  # --- 4. ANNOTATIONS ---
  geom_segment(
    data = data.frame(modality = "Language"),
    aes(x = 0.2, y = 0.2, xend = 0.28, yend = 0.3),
    arrow = arrow(length = unit(0.1, "cm")),
    color = "black", linewidth = 0.6
  ) +
  geom_text(
    data = data.frame(modality="Language"),
    aes(x = 0.25, y = 0.18),
    label = "initial",
    family = "Times", 
    fontface = "italic",
    size = 3.5
  ) +
  geom_text(
    data = data.frame(modality="Language"),
    aes(x = 0.33, y = 0.3),
    label = "final",
    family = "Times", 
    fontface = "italic",
    size = 3.5
  ) +
  # scale_y_continuous(limits = c(-0.4, 0.4)) +
  # scale_x_continuous(limits = c(-0.4, 0.4)) +
  # --- 5. THEME & FACETS ---
  facet_wrap(~modality, scales="free") + 
  theme_classic(base_size = 16, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    strip.background = element_blank(),
    strip.text.x = element_text(face='bold.italic'),
    
    # Position legend at the top and stack the two legends horizontally
    legend.position = "top",
    legend.box = "horizontal",
    
    # Target legend labels with Inconsolata
    legend.text = element_text(size = 14, face = "italic"),
    legend.box.spacing = unit(0, "pt"),
    plot.margin = margin(0, 0, 0, 0, "pt"),
    
    # --- New code for transparency and no borders ---
    plot.background = element_rect(fill = "transparent", color = NA), # Transparent canvas, no border
    panel.background = element_rect(fill = "transparent", color = NA), # Transparent plot area
    legend.background = element_rect(fill = "transparent", color = NA), # Transparent legend
    legend.box.background = element_rect(fill = "transparent", color = NA),
    panel.border = element_blank() # Removes any leftover panel borders
  ) +
  labs(
    x = "PC1",
    y = "PC2"
  )

# ggsave("figures/4b-movement-pca.pdf", width = 6.81, height = 3.9, dpi = 300, device = cairo_pdf)
ggsave("figures/4b-movement-pca-legendtop.pdf", width = 6.8, height = 4.16, dpi = 300)

# ---

wug_pca %>%
  pivot_wider(names_from = stage, values_from = c(x, y)) %>%
  mutate(
    type = case_when(
      type == "wug" ~ "sg",
      TRUE ~ "pl"
    ),
    type = factor(type, c("sg", "pl")),
    # number = factor(number, levels = c("sg", "pl"))
  ) %>%
  ggplot() + 
  
  # --- 1. REAL NOUNS ---
  geom_point(data = real_nouns, aes(x, y, color = number, shape = number), alpha = 0.08) +
  scale_color_manual(
    name = "Real Noun Type", 
    values = c("#a6611a", "#018571"),
    # Define guide directly inside the scale to avoid ggnewscale errors
    guide = guide_legend(
      title.position = "top", 
      title.hjust = 0.5, 
      override.aes = list(alpha = 1)
    )
  ) +
  scale_shape_manual(
    name = "Real Noun Type", 
    values = c(16, 17),
    guide = guide_legend(
      title.position = "top", 
      title.hjust = 0.5, 
      override.aes = list(alpha = 1)
    )
  ) +
  
  # --- 2. RESET SCALES ---
  new_scale_color() +
  
  # --- 3. NOVEL NOUNS ---
  geom_segment(
    aes(
      x = x_init, y = y_init,
      xend = x_final, yend = y_final,
      color = type
    ), 
    arrow = arrow(length = unit(0.1, "cm")),
    linewidth = 0.6, alpha = 0.6
  ) +
  scale_color_manual(
    name = "Novel Noun Type", 
    values = c("#a6611a", "#018571"),
    guide = guide_legend(
      title.position = "top", 
      title.hjust = 0.5, 
      override.aes = list(alpha = 1, linewidth = 0.6)
    )
  ) +
  
  # --- 4. ANNOTATIONS ---
  geom_segment(
    data = data.frame(modality = "Language"),
    aes(x = 0.2, y = 0.2, xend = 0.28, yend = 0.3),
    arrow = arrow(length = unit(0.1, "cm")),
    color = "black", linewidth = 0.6
  ) +
  geom_text(
    data = data.frame(modality="Language"),
    aes(x = 0.25, y = 0.18),
    label = "initial",
    family = "Times", 
    fontface = "italic",
    size = 4
  ) +
  geom_text(
    data = data.frame(modality="Language"),
    aes(x = 0.33, y = 0.3),
    label = "final",
    family = "Times", 
    fontface = "italic",
    size = 4
  ) +
  # scale_y_continuous(limits = c(-0.4, 0.4)) +
  # scale_x_continuous(limits = c(-0.4, 0.4)) +
  # --- 5. THEME & FACETS ---
  facet_wrap(~modality, scales="free", nrow = 2) + 
  theme_classic(base_size = 16, base_family = "Times") +
  theme(
    panel.grid = element_blank(),
    strip.background = element_blank(),
    strip.text.x = element_text(face='bold.italic', size = 14),
    
    # Position legend at the top and stack the two legends horizontally
    legend.position = "top",
    legend.box = "horizontal",
    
    # Target legend labels with Inconsolata
    legend.text = element_text(size = 14, face = "italic"),
    legend.box.spacing = unit(0, "pt"),
    plot.margin = margin(0, 0, 0, 0, "pt"),
    
    # --- New code for transparency and no borders ---
    plot.background = element_rect(fill = "transparent", color = NA), # Transparent canvas, no border
    panel.background = element_rect(fill = "transparent", color = NA), # Transparent plot area
    legend.background = element_rect(fill = "transparent", color = NA), # Transparent legend
    legend.box.background = element_rect(fill = "transparent", color = NA),
    panel.border = element_blank() # Removes any leftover panel borders
  ) +
  labs(
    x = "PC1",
    y = "PC2"
  )

ggsave("figures/4b-movement-pca-vertical.pdf", height = 7.95, width = 4.22, dpi = 300)
# ---

movement <- bind_rows (
  read_csv(glue("{results_dir}/wug_wugs_movement_text.csv")) %>%
    mutate(modality='Language'),
  read_csv(glue("{results_dir}/wug_wugs_movement_image.csv")) %>%
    mutate(modality='Vision')
)

# Here is your updated code. The two main changes are explicitly setting the dodge.width to match between the points and the summary, and adding fun.data = "mean_cl_normal" to generate the 95% confidence intervals.

# R
# Make sure the Hmisc package is installed for mean_cl_normal to work!
# install.packages("Hmisc")

movement %>% 
  group_by(modality, number) %>%
  summarize(
    n = n(),
    sd = sd(movement, na.rm = TRUE),
    mean_movement = mean(movement, na.rm = TRUE),
    # Calculate Standard Error
    se = sd / sqrt(n),
    # Calculate 95% Confidence Interval
    # 95% Confidence Interval
    ci_lower = mean_movement - qt(0.975, df = n - 1) * se,
    ci_upper = mean_movement + qt(0.975, df = n - 1) * se,
    
    # One-sample t-test (H0: mean = 0, HA: mean > 0)
    t_stat = mean_movement / se,
    p_value = pt(t_stat, df = n - 1, lower.tail = FALSE),
    
    .groups = "drop"
  )

movement %>% 
  group_by(modality, number) %>%
  summarize(
    mean_movement = mean(movement, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from = number, 
    values_from = mean_movement
  ) %>%
  # Apply formatting to all columns EXCEPT modality
  mutate(across(-modality, ~ sprintf("%.3f", .)))

movement %>% 
  group_by(modality, number) %>%
  summarize(
    mean_movement = mean(movement, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_wider(
    names_from = number, 
    values_from = mean_movement
  ) %>%
  # Format to 3 decimal places
  mutate(across(-modality, ~ sprintf("%.3f", .))) %>%
  # Output the dataframe as a LaTeX table
  kable(
    format = "latex", 
    booktabs = TRUE, # Makes it look clean and professional
    caption = "Mean Movement by Modality and Number",
    align = "lcccc"  # Left align the first column, center the rest (adjust based on your number of columns)
  )

movement %>% 
  # 1. Force the order of the columns: "sg" first, then "pl"
  mutate(number = factor(number, levels = c("sg", "pl"))) %>%
  
  group_by(modality, number) %>%
  summarize(
    n = n(),
    mean_movement = mean(movement, na.rm = TRUE),
    sd = sd(movement, na.rm = TRUE),
    me = qt(0.975, df = n - 1) * (sd / sqrt(n)),
    
    # 2. Format WITHOUT putting the numbers themselves in math mode. 
    # \\textsubscript{} keeps the subscript in standard text mode.
    # $\\pm$ is used only for the plus-minus symbol.
    cell_value = sprintf("%.3f\\textsubscript{$\\pm$ %.3f}", mean_movement, me),
    
    .groups = "drop"
  ) %>%
  select(modality, number, cell_value) %>%
  
  # 3. Pivot (this will automatically respect the "sg", "pl" factor order)
  pivot_wider(
    names_from = number, 
    values_from = cell_value
  ) %>%
  
  # 4. Generate the LaTeX table
  kable(
    format = "latex", 
    booktabs = TRUE, 
    escape = FALSE, # Required so kable doesn't break the LaTeX commands
    caption = "Mean Movement (with 95\\% CI Margin of Error)"
  )

movement %>%
  mutate(number = factor(number, levels = c("sg", "pl"))) %>%
  ggplot(aes(number, movement, color = modality, group = modality, shape = modality)) +
  # 1. Explicitly set dodge.width to 0.75 (the standard) so we can match it later
  geom_point(position = position_jitterdodge(dodge.width = 0.75, seed = 1024), alpha = 0.1) +
  # 2. Match the width (0.75) and use mean_cl_normal for 95% CI
  stat_summary(
    fun.data = "mean_cl_normal", 
    geom = "pointrange",
    position = position_dodge(width = 0.75),
    size = 0.4 # Optional: makes the pointrange slightly thicker to stand out
  ) +
  geom_hline(yintercept = 0.0, linetype = "dashed") +
  scale_y_continuous(limits = c(0, 0.2)) +
  theme_classic(base_size = 16, base_family = "Times") +
  labs(
    x = "Number",
    y = "Movement towards\ndesired region",
    color = "Cue Condition",
    shape = "Cue Condition"
  ) +
  theme(
    legend.position = "inside",
    legend.position.inside = c(0.95, 0.95),
    legend.justification = c("right", "top")
  )

