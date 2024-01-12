using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;
using UnityEngine.UIElements.Experimental;

//using static UnityEditor.Progress;

public class ReaderCSV : MonoBehaviour
{
    public List<string> data = new List<string>();

    Material material;


    private void Start()
    {
        ReadFile();
    }

    public void ReadFile()
    {
        material = (Material)Resources.Load("Materials/blue");
        //material = AssetDatabase.LoadAssetAtPath("Assets/Materials/blue.mat",typeof(Material)) as Material;


        CSVToolKit.LoadFile(Application.streamingAssetsPath + "/starsheet.csv", a =>
        {
            for (int j = 1;j < 70; j++) { 
                var row = a[j];
                for (int i = 0; i < row.Length; i++)
                {
                    data.Add(row[i]);
                    
                }
                row = null;
            string tag = data[0];
            float r = float.Parse(data[1]);
            float x = float.Parse(data[2]);
            float y = float.Parse(data[3]);
            float z = float.Parse(data[4]);

                data.Clear();


            GameObject obj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
               // obj.gameObject.AddComponent<MeshRenderer>();
                obj.gameObject.AddComponent<BoxCollider>();
                
                obj.GetComponent<Renderer>().material = material;
               // material.color = Color.yellow;
                obj.transform.position = new Vector3(x, y, z);
                obj.transform.localScale += new Vector3(r, r, r);
                obj.name = tag;
           


             
            }

        });
    

}







}
