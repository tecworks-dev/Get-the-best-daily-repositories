/*
 * Designed and developed by 2024 skydoves (Jaewoong Eum)
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
package com.skydoves.firebase.database.ktx

import com.google.firebase.database.DatabaseError

sealed class ChildState<T : Any> {

  data class ChildAdded<T : Any>(
    val value: T?,
    val previousChildName: String?,
  ) : ChildState<T>()

  data class ChildChanged<T : Any>(
    val value: T?,
    val previousChildName: String?,
  ) : ChildState<T>()

  data class ChildRemoved<T : Any>(
    val value: T?,
  ) : ChildState<T>()

  data class ChildMoved<T : Any>(
    val value: T?,
    val previousChildName: String?,
  ) : ChildState<T>()

  data class ChildCanceled<T : Any>(
    val error: DatabaseError,
  ) : ChildState<T>()
}
