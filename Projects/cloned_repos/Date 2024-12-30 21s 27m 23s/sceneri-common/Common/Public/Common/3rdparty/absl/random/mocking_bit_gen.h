// Copyright 2018 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// -----------------------------------------------------------------------------
// mocking_bit_gen.h
// -----------------------------------------------------------------------------
//
// This file includes an `absl::MockingBitGen` class to use as a mock within the
// Googletest testing framework. Such a mock is useful to provide deterministic
// values as return values within (otherwise random) Abseil distribution
// functions. Such determinism within a mock is useful within testing frameworks
// to test otherwise indeterminate APIs.
//
// More information about the Googletest testing framework is available at
// https://github.com/google/googletest

#ifndef ABSL_RANDOM_MOCKING_BIT_GEN_H_
#define ABSL_RANDOM_MOCKING_BIT_GEN_H_

#include <iterator>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "Common/3rdparty/absl/base/internal/fast_type_id.h"
#include "Common/3rdparty/absl/container/flat_hash_map.h"
#include "Common/3rdparty/absl/meta/type_traits.h"
#include "Common/3rdparty/absl/random/distributions.h"
#include "Common/3rdparty/absl/random/internal/distribution_caller.h"
#include "Common/3rdparty/absl/random/random.h"
#include "Common/3rdparty/absl/strings/str_cat.h"
#include "Common/3rdparty/absl/strings/str_join.h"
#include "Common/3rdparty/absl/types/span.h"
#include "Common/3rdparty/absl/types/variant.h"
#include "Common/3rdparty/absl/utility/utility.h"

namespace absl {
ABSL_NAMESPACE_BEGIN

namespace random_internal {
template <typename>
struct DistributionCaller;
class MockHelpers;

}  // namespace random_internal
class BitGenRef;

// MockingBitGen
//
// `absl::MockingBitGen` is a mock Uniform Random Bit Generator (URBG) class
// which can act in place of an `absl::BitGen` URBG within tests using the
// Googletest testing framework.
//
// Usage:
//
// Use an `absl::MockingBitGen` along with a mock distribution object (within
// mock_distributions.h) inside Googletest constructs such as ON_CALL(),
// EXPECT_TRUE(), etc. to produce deterministic results conforming to the
// distribution's API contract.
//
// Example:
//
//  // Mock a call to an `absl::Bernoulli` distribution using Googletest
//   absl::MockingBitGen bitgen;
//
//   ON_CALL(absl::MockBernoulli(), Call(bitgen, 0.5))
//       .WillByDefault(testing::Return(true));
//   EXPECT_TRUE(absl::Bernoulli(bitgen, 0.5));
//
//  // Mock a call to an `absl::Uniform` distribution within Googletest
//  absl::MockingBitGen bitgen;
//
//   ON_CALL(absl::MockUniform<int>(), Call(bitgen, testing::_, testing::_))
//       .WillByDefault([] (int low, int high) {
//           return (low + high) / 2;
//       });
//
//   EXPECT_EQ(absl::Uniform<int>(gen, 0, 10), 5);
//   EXPECT_EQ(absl::Uniform<int>(gen, 30, 40), 35);
//
// At this time, only mock distributions supplied within the Abseil random
// library are officially supported.
//
// EXPECT_CALL and ON_CALL need to be made within the same DLL component as
// the call to absl::Uniform and related methods, otherwise mocking will fail
// since the  underlying implementation creates a type-specific pointer which
// will be distinct across different DLL boundaries.
//
class MockingBitGen {
 public:
  MockingBitGen() = default;

  ~MockingBitGen() {
    for (const auto& del : deleters_) del();
  }

  // URBG interface
  using result_type = absl::BitGen::result_type;

  static constexpr result_type(min)() { return (absl::BitGen::min)(); }
  static constexpr result_type(max)() { return (absl::BitGen::max)(); }
  result_type operator()() { return gen_(); }

 private:
  using match_impl_fn = void (*)(void* mock_fn, void* t_erased_arg_tuple,
                                 void* t_erased_result);

  struct MockData {
    void* mock_fn = nullptr;
    match_impl_fn match_impl = nullptr;
  };

  // GetMockFnType returns the testing::MockFunction for a result and tuple.
  // This method only exists for type deduction and is otherwise unimplemented.
  template <typename ResultT, typename... Args>
  static auto GetMockFnType(ResultT, std::tuple<Args...>)
      -> ::testing::MockFunction<ResultT(Args...)>;

  // MockFnCaller is a helper method for use with absl::apply to
  // apply an ArgTupleT to a compatible MockFunction.
  // NOTE: MockFnCaller is essentially equivalent to the lambda:
  // [fn](auto... args) { return fn->Call(std::move(args)...)}
  // however that fails to build on some supported platforms.
  template <typename ResultT, typename MockFnType, typename Tuple>
  struct MockFnCaller;
  // specialization for std::tuple.
  template <typename ResultT, typename MockFnType, typename... Args>
  struct MockFnCaller<ResultT, MockFnType, std::tuple<Args...>> {
    MockFnType* fn;
    inline ResultT operator()(Args... args) {
      return fn->Call(std::move(args)...);
    }
  };

  // MockingBitGen::RegisterMock
  //
  // RegisterMock<ResultT, ArgTupleT>(FastTypeIdType) is the main extension
  // point for extending the MockingBitGen framework. It provides a mechanism to
  // install a mock expectation for a function like ResultT(Args...) keyed by
  // type_idex onto the MockingBitGen context. The key is that the type_index
  // used to register must match the type index used to call the mock.
  //
  // The returned MockFunction<...> type can be used to setup additional
  // distribution parameters of the expectation.
  template <typename ResultT, typename ArgTupleT>
  auto RegisterMock(base_internal::FastTypeIdType type)
      -> decltype(GetMockFnType(std::declval<ResultT>(),
                                std::declval<ArgTupleT>()))& {
    using MockFnType = decltype(
        GetMockFnType(std::declval<ResultT>(), std::declval<ArgTupleT>()));
    auto& mock = mocks_[type];
    if (!mock.mock_fn) {
      auto* mock_fn = new MockFnType;
      mock.mock_fn = mock_fn;
      mock.match_impl = &MatchImpl<ResultT, ArgTupleT>;
      deleters_.emplace_back([mock_fn] { delete mock_fn; });
    }
    return *static_cast<MockFnType*>(mock.mock_fn);
  }

  // MockingBitGen::MatchImpl<> is a dispatch function which converts the
  // generic type-erased parameters into a specific mock invocation call.
  // Requires tuple_args to point to a ArgTupleT, which is a std::tuple<Args...>
  // used to invoke the mock function.
  // Requires result to point to a ResultT, which is the result of the call.
  template <typename ResultT, typename ArgTupleT>
  static void MatchImpl(/*MockFnType<ResultT, Args...>*/ void* mock_fn,
                        /*ArgTupleT*/ void* args_tuple,
                        /*ResultT*/ void* result) {
    using MockFnType = decltype(
        GetMockFnType(std::declval<ResultT>(), std::declval<ArgTupleT>()));
    *static_cast<ResultT*>(result) = absl::apply(
        MockFnCaller<ResultT, MockFnType, ArgTupleT>{
            static_cast<MockFnType*>(mock_fn)},
        *static_cast<ArgTupleT*>(args_tuple));
  }

  // MockingBitGen::InvokeMock
  //
  // InvokeMock(FastTypeIdType, args, result) is the entrypoint for invoking
  // mocks registered on MockingBitGen.
  //
  // When no mocks are registered on the provided FastTypeIdType, returns false.
  // Otherwise attempts to invoke the mock function ResultT(Args...) that
  // was previously registered via the type_index.
  // Requires tuple_args to point to a ArgTupleT, which is a std::tuple<Args...>
  // used to invoke the mock function.
  // Requires result to point to a ResultT, which is the result of the call.
  inline bool InvokeMock(base_internal::FastTypeIdType type, void* args_tuple,
                         void* result) {
    // Trigger a mock, if there exists one that matches `param`.
    auto it = mocks_.find(type);
    if (it == mocks_.end()) return false;
    auto* mock_data = static_cast<MockData*>(&it->second);
    mock_data->match_impl(mock_data->mock_fn, args_tuple, result);
    return true;
  }

  absl::flat_hash_map<base_internal::FastTypeIdType, MockData> mocks_;
  std::vector<std::function<void()>> deleters_;
  absl::BitGen gen_;

  template <typename>
  friend struct ::absl::random_internal::DistributionCaller;  // for InvokeMock
  friend class ::absl::BitGenRef;                             // for InvokeMock
  friend class ::absl::random_internal::MockHelpers;  // for RegisterMock,
                                                      // InvokeMock
};

ABSL_NAMESPACE_END
}  // namespace absl

#endif  // ABSL_RANDOM_MOCKING_BIT_GEN_H_
