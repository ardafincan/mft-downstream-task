use pyo3::prelude::*;
use crate::tokenizer::TurkishTokenizer;

mod decoder;
mod tokenizer;

#[pyclass(name = "TurkishTokenizer")]
struct PyTurkishTokenizer {
    inner: TurkishTokenizer,
}

#[pymethods]
impl PyTurkishTokenizer {
    #[new]
    fn new() -> PyResult<Self> {
        // Embed the JSON files into the binary
        let roots_json = include_str!("resources/kokler.json");
        let ekler_json = include_str!("resources/ekler.json");
        let bpe_json = include_str!("resources/bpe_tokenler.json");
        
        let inner = TurkishTokenizer::from_files(roots_json, ekler_json, bpe_json)
             .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
             
        Ok(PyTurkishTokenizer { inner })
    }
    
    fn encode(&self, text: &str) -> Vec<i32> {
        self.inner.encode(text)
    }
    
    fn decode(&self, ids: Vec<i32>) -> String {
        self.inner.decode(ids)
    }
}

#[pymodule]
fn turk_mft(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTurkishTokenizer>()?;
    Ok(())
}
