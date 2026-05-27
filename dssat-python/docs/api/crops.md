# Crop Models

## Base Class

::: dssat.crop.base.CropModel

---

## CERES-Maize

Full implementation of the **CERES-Maize** model from `Plant/CERES-Maize/`.  Supports all 9 growth stages (sowing → harvest), RUE-based photosynthesis, per-stage biomass partitioning, and crown-temperature thermal time with snow correction.

### Main Model

::: dssat.crop.ceres_maize.ceres_maize.CeresMaize

### Cultivar Coefficients

::: dssat.crop.ceres_maize.ceres_maize.MaizeCultivar

### Phenology

::: dssat.crop.ceres_maize.phenology

### Growth

::: dssat.crop.ceres_maize.growth

### Roots

::: dssat.crop.ceres_maize.roots

---

## CERES-Wheat

Full implementation of the **CERES-Wheat** model.

### Main Model

::: dssat.crop.ceres_wheat.ceres_wheat.CeresWheat

### Cultivar Coefficients

::: dssat.crop.ceres_wheat.phenology.WheatCultivar

### Phenology

::: dssat.crop.ceres_wheat.phenology.WheatPhenology

### Growth

::: dssat.crop.ceres_wheat.growth.WheatGrowth

---

## CERES-Sorghum

Full implementation of the **CERES-Sorghum** model.

### Main Model

::: dssat.crop.ceres_sorghum.ceres_sorghum.CeresSorghum

### Cultivar Coefficients

::: dssat.crop.ceres_sorghum.phenology.SorghumCultivar

### Phenology

::: dssat.crop.ceres_sorghum.phenology.SorghumPhenology

### Growth

::: dssat.crop.ceres_sorghum.growth.SorghumGrowth

---

## CERES-Millet

Full implementation of the **CERES-Millet** model.

### Main Model

::: dssat.crop.ceres_millet.ceres_millet.CeresMillet

### Cultivar Coefficients

::: dssat.crop.ceres_millet.phenology.MilletCultivar

---

## CROPGRO

Full implementation of the **CROPGRO** model framework, supporting soybean and spring canola.

### Soybean Model

::: dssat.crop.cropgro.soybean.CropGROSoybean

### Soybean Cultivar Coefficients

::: dssat.crop.cropgro.soybean.SoybeanCultivar

### Canola Model

::: dssat.crop.cropgro.canola.CropGROCanola

### Canola Cultivar Coefficients

::: dssat.crop.cropgro.canola.CanolaCultivar

### Phenology

::: dssat.crop.cropgro.phenology.CropGROPhenology

### Growth

::: dssat.crop.cropgro.growth.CropGROGrowth

### Nitrogen Fixation

::: dssat.crop.cropgro.nfix.CropGRONFix

---

## Other Crop Models *(stubs)*

The following models are planned.  Their `__init__.py` files document the expected interface.

::: dssat.crop.ceres_rice

::: dssat.crop.substor_potato

::: dssat.crop.canegro

::: dssat.crop.cscas_cassava

::: dssat.crop.samuca

::: dssat.crop.ceres_sugarbeet
