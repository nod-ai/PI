//
// Created by maksim on 6/13/23.
//

#ifndef PI_TORCHDTYPE_H
#define PI_TORCHDTYPE_H

#include <type_traits>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace mlir::torch {

enum DType {
  //  |-------------------|--------------------|
  //  | Torch Type        | MLIR Type          |
  //  |-------------------|--------------------|
  //  | torch.bfloat16    | bf16               |
  //  | torch.bool        | i1                 |
  //  | torch.complex*    | complex<*>         |
  //  | torch.float16     | f16                |
  //  | torch.float32     | f32                |
  //  | torch.float64     | f64                |
  //  | torch.int16       | si16               |
  //  | torch.int32       | si32               |
  //  | torch.int64       | si64               |
  //  | torch.int8        | si8                |
  //  | torch.qint8       | !torch.qint8       |
  //  | torch.quint8      | !torch.quint8      |
  //  | torch.uint8       | ui8                |
  //  |-------------------|--------------------|
  uint8 = 0,
  int8 = 1,
  int16 = 2,
  int32 = 3,
  int64 = 4,
  float16 = 5,
  float32 = 6,
  float64 = 7,
  // complex_half 8
  complex32 = 9,
  complex64 = 10,
  bool_ = 11,
  qint8 = 12,
  quint8 = 13,
  // qint32 14
  bfloat16 = 15
};

template <typename E> constexpr auto to_underlying(E e) noexcept {
  return static_cast<std::underlying_type_t<E>>(e);
}

void populateTorchDType(py::module &m);

} // namespace mlir::torch
#endif // PI_TORCHDTYPE_H
