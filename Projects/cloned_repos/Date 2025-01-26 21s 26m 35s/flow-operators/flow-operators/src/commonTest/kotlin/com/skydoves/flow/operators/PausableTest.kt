/*
 * Designed and developed by 2025 skydoves (Jaewoong Eum)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.skydoves.flow.operators

import app.cash.turbine.test
import com.skydoves.flow.operators.pausable.PausableStateFlow
import com.skydoves.flow.operators.pausable.pausableStateIn
import kotlinx.coroutines.cancelChildren
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.job
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals

class PausableTest {

  @Test
  fun `pausableFlow should paused and resumed the upstream cold flow`() = runTest {
    var restartCount = 0

    val restartableStateFlow: PausableStateFlow<String> = flow {
      if (restartCount == 0) {
        emit("aaa")
      } else {
        emit("bbb")
      }
      restartCount++
    }.pausableStateIn(
      scope = backgroundScope,
      started = SharingStarted.Lazily,
      initialValue = "initial",
    )

    restartableStateFlow.test {
      assertEquals(0, restartCount)
      assertEquals("initial", awaitItem())
      assertEquals("aaa", awaitItem())

      restartableStateFlow.pause()
      restartableStateFlow.resume()

      assertEquals(1, restartCount)
      assertEquals("bbb", awaitItem())

      restartableStateFlow.pause()

      expectNoEvents()
    }

    coroutineContext.job.cancelChildren()
  }
}
