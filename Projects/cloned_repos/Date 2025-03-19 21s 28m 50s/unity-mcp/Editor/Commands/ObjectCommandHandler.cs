using UnityEngine;
using Newtonsoft.Json.Linq;
using MCPServer.Editor.Helpers;
using System;
using System.Linq;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine.SceneManagement;
using UnityEditor.SceneManagement;

namespace MCPServer.Editor.Commands
{
    /// <summary>
    /// Handles object-related commands
    /// </summary>
    public static class ObjectCommandHandler
    {
        /// <summary>
        /// Gets information about a specific object
        /// </summary>
        public static object GetObjectInfo(JObject @params)
        {
            string name = (string)@params["name"] ?? throw new System.Exception("Parameter 'name' is required.");
            var obj = GameObject.Find(name) ?? throw new System.Exception($"Object '{name}' not found.");
            return new
            {
                name = obj.name,
                position = new[] { obj.transform.position.x, obj.transform.position.y, obj.transform.position.z },
                rotation = new[] { obj.transform.eulerAngles.x, obj.transform.eulerAngles.y, obj.transform.eulerAngles.z },
                scale = new[] { obj.transform.localScale.x, obj.transform.localScale.y, obj.transform.localScale.z }
            };
        }

        /// <summary>
        /// Creates a new object in the scene
        /// </summary>
        public static object CreateObject(JObject @params)
        {
            string type = (string)@params["type"] ?? throw new System.Exception("Parameter 'type' is required.");
            GameObject obj = type.ToUpper() switch
            {
                "CUBE" => GameObject.CreatePrimitive(PrimitiveType.Cube),
                "SPHERE" => GameObject.CreatePrimitive(PrimitiveType.Sphere),
                "CYLINDER" => GameObject.CreatePrimitive(PrimitiveType.Cylinder),
                "CAPSULE" => GameObject.CreatePrimitive(PrimitiveType.Capsule),
                "PLANE" => GameObject.CreatePrimitive(PrimitiveType.Plane),
                "EMPTY" => new GameObject(),
                "CAMERA" => new GameObject("Camera") { }.AddComponent<Camera>().gameObject,
                "LIGHT" => new GameObject("Light") { }.AddComponent<Light>().gameObject,
                "DIRECTIONAL_LIGHT" => CreateDirectionalLight(),
                _ => throw new System.Exception($"Unsupported object type: {type}")
            };

            if (@params.ContainsKey("name")) obj.name = (string)@params["name"];
            if (@params.ContainsKey("location")) obj.transform.position = Vector3Helper.ParseVector3((JArray)@params["location"]);
            if (@params.ContainsKey("rotation")) obj.transform.eulerAngles = Vector3Helper.ParseVector3((JArray)@params["rotation"]);
            if (@params.ContainsKey("scale")) obj.transform.localScale = Vector3Helper.ParseVector3((JArray)@params["scale"]);

            return new { name = obj.name };
        }

        /// <summary>
        /// Modifies an existing object's properties
        /// </summary>
        public static object ModifyObject(JObject @params)
        {
            string name = (string)@params["name"] ?? throw new System.Exception("Parameter 'name' is required.");
            var obj = GameObject.Find(name) ?? throw new System.Exception($"Object '{name}' not found.");

            // Handle basic transform properties
            if (@params.ContainsKey("location")) obj.transform.position = Vector3Helper.ParseVector3((JArray)@params["location"]);
            if (@params.ContainsKey("rotation")) obj.transform.eulerAngles = Vector3Helper.ParseVector3((JArray)@params["rotation"]);
            if (@params.ContainsKey("scale")) obj.transform.localScale = Vector3Helper.ParseVector3((JArray)@params["scale"]);
            if (@params.ContainsKey("visible")) obj.SetActive((bool)@params["visible"]);

            // Handle parent setting
            if (@params.ContainsKey("set_parent"))
            {
                string parentName = (string)@params["set_parent"];
                var parent = GameObject.Find(parentName) ?? throw new System.Exception($"Parent object '{parentName}' not found.");
                obj.transform.SetParent(parent.transform);
            }

            // Handle component operations
            if (@params.ContainsKey("add_component"))
            {
                string componentType = (string)@params["add_component"];
                Type type = componentType switch
                {
                    "Rigidbody" => typeof(Rigidbody),
                    "BoxCollider" => typeof(BoxCollider),
                    "SphereCollider" => typeof(SphereCollider),
                    "CapsuleCollider" => typeof(CapsuleCollider),
                    "MeshCollider" => typeof(MeshCollider),
                    "Camera" => typeof(Camera),
                    "Light" => typeof(Light),
                    "Renderer" => typeof(Renderer),
                    "MeshRenderer" => typeof(MeshRenderer),
                    "SkinnedMeshRenderer" => typeof(SkinnedMeshRenderer),
                    "Animator" => typeof(Animator),
                    "AudioSource" => typeof(AudioSource),
                    "AudioListener" => typeof(AudioListener),
                    "ParticleSystem" => typeof(ParticleSystem),
                    "ParticleSystemRenderer" => typeof(ParticleSystemRenderer),
                    "TrailRenderer" => typeof(TrailRenderer),
                    "LineRenderer" => typeof(LineRenderer),
                    "TextMesh" => typeof(TextMesh),
                    "TextMeshPro" => typeof(TMPro.TextMeshPro),
                    "TextMeshProUGUI" => typeof(TMPro.TextMeshProUGUI),
                    _ => Type.GetType($"UnityEngine.{componentType}") ??
                         Type.GetType(componentType) ??
                         throw new System.Exception($"Component type '{componentType}' not found.")
                };
                obj.AddComponent(type);
            }

            if (@params.ContainsKey("remove_component"))
            {
                string componentType = (string)@params["remove_component"];
                Type type = Type.GetType($"UnityEngine.{componentType}") ??
                           Type.GetType(componentType) ??
                           throw new System.Exception($"Component type '{componentType}' not found.");
                var component = obj.GetComponent(type);
                if (component != null)
                    UnityEngine.Object.DestroyImmediate(component);
            }

            // Handle property setting
            if (@params.ContainsKey("set_property"))
            {
                var propertyData = (JObject)@params["set_property"];
                string componentType = (string)propertyData["component"];
                string propertyName = (string)propertyData["property"];
                var value = propertyData["value"];

                // Handle GameObject properties separately
                if (componentType == "GameObject")
                {
                    var gameObjectProperty = typeof(GameObject).GetProperty(propertyName) ??
                                 throw new System.Exception($"Property '{propertyName}' not found on GameObject.");

                    // Convert value based on property type
                    object gameObjectValue = Convert.ChangeType(value, gameObjectProperty.PropertyType);
                    gameObjectProperty.SetValue(obj, gameObjectValue);
                    return new { name = obj.name };
                }

                // Handle component properties
                Type type = componentType switch
                {
                    "Rigidbody" => typeof(Rigidbody),
                    "BoxCollider" => typeof(BoxCollider),
                    "SphereCollider" => typeof(SphereCollider),
                    "CapsuleCollider" => typeof(CapsuleCollider),
                    "MeshCollider" => typeof(MeshCollider),
                    "Camera" => typeof(Camera),
                    "Light" => typeof(Light),
                    "Renderer" => typeof(Renderer),
                    "MeshRenderer" => typeof(MeshRenderer),
                    "SkinnedMeshRenderer" => typeof(SkinnedMeshRenderer),
                    "Animator" => typeof(Animator),
                    "AudioSource" => typeof(AudioSource),
                    "AudioListener" => typeof(AudioListener),
                    "ParticleSystem" => typeof(ParticleSystem),
                    "ParticleSystemRenderer" => typeof(ParticleSystemRenderer),
                    "TrailRenderer" => typeof(TrailRenderer),
                    "LineRenderer" => typeof(LineRenderer),
                    "TextMesh" => typeof(TextMesh),
                    "TextMeshPro" => typeof(TMPro.TextMeshPro),
                    "TextMeshProUGUI" => typeof(TMPro.TextMeshProUGUI),
                    _ => Type.GetType($"UnityEngine.{componentType}") ??
                         Type.GetType(componentType) ??
                         throw new System.Exception($"Component type '{componentType}' not found.")
                };

                var component = obj.GetComponent(type) ??
                               throw new System.Exception($"Component '{componentType}' not found on object '{name}'.");

                var property = type.GetProperty(propertyName) ??
                              throw new System.Exception($"Property '{propertyName}' not found on component '{componentType}'.");

                // Convert value based on property type
                object propertyValue = Convert.ChangeType(value, property.PropertyType);
                property.SetValue(component, propertyValue);
            }

            return new { name = obj.name };
        }

        /// <summary>
        /// Deletes an object from the scene
        /// </summary>
        public static object DeleteObject(JObject @params)
        {
            string name = (string)@params["name"] ?? throw new System.Exception("Parameter 'name' is required.");
            var obj = GameObject.Find(name) ?? throw new System.Exception($"Object '{name}' not found.");
            UnityEngine.Object.DestroyImmediate(obj);
            return new { name };
        }

        /// <summary>
        /// Gets all properties of a specified game object
        /// </summary>
        public static object GetObjectProperties(JObject @params)
        {
            string name = (string)@params["name"] ?? throw new System.Exception("Parameter 'name' is required.");
            var obj = GameObject.Find(name) ?? throw new System.Exception($"Object '{name}' not found.");

            var components = obj.GetComponents<Component>()
                .Select(c => new
                {
                    type = c.GetType().Name,
                    properties = GetComponentProperties(c)
                })
                .ToList();

            return new
            {
                name = obj.name,
                tag = obj.tag,
                layer = obj.layer,
                active = obj.activeSelf,
                transform = new
                {
                    position = new[] { obj.transform.position.x, obj.transform.position.y, obj.transform.position.z },
                    rotation = new[] { obj.transform.eulerAngles.x, obj.transform.eulerAngles.y, obj.transform.eulerAngles.z },
                    scale = new[] { obj.transform.localScale.x, obj.transform.localScale.y, obj.transform.localScale.z }
                },
                components
            };
        }

        /// <summary>
        /// Gets properties of a specific component
        /// </summary>
        public static object GetComponentProperties(JObject @params)
        {
            string objectName = (string)@params["object_name"] ?? throw new System.Exception("Parameter 'object_name' is required.");
            string componentType = (string)@params["component_type"] ?? throw new System.Exception("Parameter 'component_type' is required.");

            var obj = GameObject.Find(objectName) ?? throw new System.Exception($"Object '{objectName}' not found.");
            var component = obj.GetComponent(componentType) ?? throw new System.Exception($"Component '{componentType}' not found on object '{objectName}'.");

            return GetComponentProperties(component);
        }

        /// <summary>
        /// Finds objects by name in the scene
        /// </summary>
        public static object FindObjectsByName(JObject @params)
        {
            string name = (string)@params["name"] ?? throw new System.Exception("Parameter 'name' is required.");
            var objects = GameObject.FindObjectsByType<GameObject>(FindObjectsSortMode.None)
                .Where(o => o.name.Contains(name))
                .Select(o => new
                {
                    name = o.name,
                    path = GetGameObjectPath(o)
                })
                .ToList();

            return new { objects };
        }

        /// <summary>
        /// Finds objects by tag in the scene
        /// </summary>
        public static object FindObjectsByTag(JObject @params)
        {
            string tag = (string)@params["tag"] ?? throw new System.Exception("Parameter 'tag' is required.");
            var objects = GameObject.FindGameObjectsWithTag(tag)
                .Select(o => new
                {
                    name = o.name,
                    path = GetGameObjectPath(o)
                })
                .ToList();

            return new { objects };
        }

        /// <summary>
        /// Gets the current hierarchy of game objects in the scene
        /// </summary>
        public static object GetHierarchy()
        {
            var rootObjects = SceneManager.GetActiveScene().GetRootGameObjects();
            var hierarchy = rootObjects.Select(o => BuildHierarchyNode(o)).ToList();

            return new { hierarchy };
        }

        /// <summary>
        /// Selects a specified game object in the editor
        /// </summary>
        public static object SelectObject(JObject @params)
        {
            string name = (string)@params["name"] ?? throw new System.Exception("Parameter 'name' is required.");
            var obj = GameObject.Find(name) ?? throw new System.Exception($"Object '{name}' not found.");

            Selection.activeGameObject = obj;
            return new { name = obj.name };
        }

        /// <summary>
        /// Gets the currently selected game object in the editor
        /// </summary>
        public static object GetSelectedObject()
        {
            var selected = Selection.activeGameObject;
            if (selected == null)
                return new { selected = (object)null };

            return new
            {
                selected = new
                {
                    name = selected.name,
                    path = GetGameObjectPath(selected)
                }
            };
        }

        // Helper methods
        private static Dictionary<string, object> GetComponentProperties(Component component)
        {
            var properties = new Dictionary<string, object>();
            var serializedObject = new SerializedObject(component);
            var property = serializedObject.GetIterator();

            while (property.Next(true))
            {
                properties[property.name] = GetPropertyValue(property);
            }

            return properties;
        }

        private static object GetPropertyValue(SerializedProperty property)
        {
            switch (property.propertyType)
            {
                case SerializedPropertyType.Integer:
                    return property.intValue;
                case SerializedPropertyType.Float:
                    return property.floatValue;
                case SerializedPropertyType.Boolean:
                    return property.boolValue;
                case SerializedPropertyType.String:
                    return property.stringValue;
                case SerializedPropertyType.Vector3:
                    return new[] { property.vector3Value.x, property.vector3Value.y, property.vector3Value.z };
                case SerializedPropertyType.Vector2:
                    return new[] { property.vector2Value.x, property.vector2Value.y };
                case SerializedPropertyType.Color:
                    return new[] { property.colorValue.r, property.colorValue.g, property.colorValue.b, property.colorValue.a };
                case SerializedPropertyType.ObjectReference:
                    return property.objectReferenceValue ? property.objectReferenceValue.name : null;
                default:
                    return property.propertyType.ToString();
            }
        }

        private static string GetGameObjectPath(GameObject obj)
        {
            var path = obj.name;
            var parent = obj.transform.parent;

            while (parent != null)
            {
                path = parent.name + "/" + path;
                parent = parent.parent;
            }

            return path;
        }

        private static object BuildHierarchyNode(GameObject obj)
        {
            return new
            {
                name = obj.name,
                children = Enumerable.Range(0, obj.transform.childCount)
                    .Select(i => BuildHierarchyNode(obj.transform.GetChild(i).gameObject))
                    .ToList()
            };
        }

        /// <summary>
        /// Creates a directional light game object
        /// </summary>
        private static GameObject CreateDirectionalLight()
        {
            var obj = new GameObject("DirectionalLight");
            var light = obj.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.0f;
            light.shadows = LightShadows.Soft;
            return obj;
        }
    }
}