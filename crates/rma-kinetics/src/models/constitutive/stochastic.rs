use derive_builder::Builder;
use differential_equations::sde::SDE;
use rand::{SeedableRng, rngs::StdRng};

use rand_distr::{Distribution as _, Normal};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::State;

const DEFAULT_PROD: f64 = 0.2;
const DEFAULT_BBB_TRANSPORT: f64 = 0.6;
const DEFAULT_DEG: f64 = 0.007;
const DEFAULT_PROD_STDV: f64 = 0.5;
const DEFAULT_TRANSPORT_STDV: f64 = 0.1;
const DEFAULT_SEED: u64 = 42;

fn rng_from_seed(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

/// Stochastic constitutive RMA expression model.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(from = "ModelSerde", into = "ModelSerde"))]
#[derive(Builder, Clone)]
#[builder(build_fn(private, name = "build_internal"), derive(Debug))]
pub struct Model {
    /// RMA production rate.
    #[builder(default = "DEFAULT_PROD")]
    pub prod: f64,
    /// RMA blood-brain barrier transport rate.
    #[builder(default = "DEFAULT_BBB_TRANSPORT")]
    pub bbb_transport: f64,
    /// RMA degradation rate.
    #[builder(default = "DEFAULT_DEG")]
    pub deg: f64,
    /// Gaussian noise standard deviation of protein production and secretion.
    #[builder(default = "DEFAULT_PROD_STDV")]
    pub prod_noise: f64,
    #[builder(default = "DEFAULT_TRANSPORT_STDV")]
    pub transport_noise: f64,
    /// Random seed used to initialize the RNG.
    #[builder(default = "DEFAULT_SEED")]
    pub seed: u64,
    /// Random number generator.
    #[builder(setter(skip), default = "rng_from_seed(DEFAULT_SEED)")]
    pub(crate) rng: StdRng,
}

impl Model {
    /// Create a new stochastic constitutive expression model.
    pub fn new(
        prod: f64,
        bbb_transport: f64,
        deg: f64,
        prod_noise: f64,
        transport_noise: f64,
        seed: u64,
    ) -> Self {
        Self {
            prod,
            bbb_transport,
            deg,
            prod_noise,
            transport_noise,
            seed,
            rng: rng_from_seed(seed),
        }
    }

    /// Create a new ModelBuilder for constructing a model instance.
    pub fn builder() -> ModelBuilder {
        ModelBuilder::default()
    }

    /// Set a new random seed and reinitialize the internal RNG.
    pub fn reseed(&mut self, seed: u64) {
        self.seed = seed;
        self.rng = rng_from_seed(seed);
    }
}

impl Default for Model {
    fn default() -> Self {
        ModelBuilder::default().build().unwrap()
    }
}

impl ModelBuilder {
    pub fn build(&self) -> Result<Model, ModelBuilderError> {
        let mut model = self.build_internal()?;
        model.rng = rng_from_seed(model.seed);
        Ok(model)
    }
}

#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
struct ModelSerde {
    prod: f64,
    bbb_transport: f64,
    deg: f64,
    prod_noise: f64,
    transport_noise: f64,
    seed: u64,
}

#[cfg(feature = "serde")]
impl From<ModelSerde> for Model {
    fn from(value: ModelSerde) -> Self {
        Model::new(
            value.prod,
            value.bbb_transport,
            value.deg,
            value.prod_noise,
            value.transport_noise,
            value.seed,
        )
    }
}

#[cfg(feature = "serde")]
impl From<Model> for ModelSerde {
    fn from(value: Model) -> Self {
        Self {
            prod: value.prod,
            bbb_transport: value.bbb_transport,
            deg: value.deg,
            prod_noise: value.prod_noise,
            transport_noise: value.transport_noise,
            seed: value.seed,
        }
    }
}

impl SDE<f64, State<f64>> for Model {
    /// Deterministic drift term for constitutive RMA expression.
    fn drift(&self, _t: f64, y: &State<f64>, dydt: &mut State<f64>) {
        let brain_efflux = self.bbb_transport * y.brain_rma;
        dydt.brain_rma = self.prod - brain_efflux;
        dydt.plasma_rma = brain_efflux - (self.deg * y.plasma_rma);
    }

    /// Diffusion term for constitutive RMA expression.
    fn diffusion(&self, _t: f64, y: &State<f64>, dydw: &mut State<f64>) {
        dydw.brain_rma = self.prod_noise * y.brain_rma;
        dydw.plasma_rma = self.transport_noise * y.plasma_rma;
    }

    /// Noise term for constitutive RMA expression.
    fn noise(&mut self, dt: f64, dw: &mut State<f64>) {
        let normal = Normal::new(0.0, dt.sqrt()).unwrap();
        dw.brain_rma = normal.sample(&mut self.rng);
        dw.plasma_rma = normal.sample(&mut self.rng);
    }
}
