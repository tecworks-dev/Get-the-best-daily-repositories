/*
 * Designed and developed by 2024 skydoves (Jaewoong Eum)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.getstream.server.driven.core.data.repository

import com.google.firebase.database.DatabaseReference
import com.skydoves.firebase.database.ktx.flow
import io.getstream.server.driven.core.model.TimelineUi
import javax.inject.Inject
import kotlinx.coroutines.flow.Flow
import kotlinx.serialization.json.Json

internal class TimelineRepositoryImpl @Inject constructor(
  private val databaseReference: DatabaseReference,
  private val json: Json
) : TimelineRepository {

  override fun fetchTimelineUi(): Flow<Result<TimelineUi?>> {
    return databaseReference.flow(
      path = { snapshot ->
        snapshot.child("timeline")
      },
      decodeProvider = { jsonString ->
        json.decodeFromString(jsonString)
      }
    )
  }
}