using UnityEngine;
using UnityEditor;

[CustomPropertyDrawer(typeof(OverridePropertyAttribute))]
public class OverridePropertyDrawer : PropertyDrawer
{
    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
    {
        SerializedProperty overrideProp =
            property.serializedObject.FindProperty(property.propertyPath + "Override");
        Rect posToggle = position;
        posToggle.width = 16;
        EditorGUI.PropertyField(posToggle, overrideProp, GUIContent.none);

        using (new EditorGUI.DisabledScope(!overrideProp.boolValue))
        {
            position.xMin += 16;
            EditorGUIUtility.labelWidth -= 16;
            EditorGUI.PropertyField(position, property, label);
            EditorGUIUtility.labelWidth = 0;
        }
    }
}
