using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using static UnityEngine.GraphicsBuffer;

public class CameraContro : MonoBehaviour
{

    public GameObject Target;
    public GameObject Target2;

    private float horiAngle;
    private float vertiAngle;

    public float distance;

    private Vector3 tem;

    public float mScrollValue = 10;

    private CharacterController cc;
    private float horizontalMove, verticalMove, shangxiaMove;
    public float moveSpeed;
    private Vector3 fangxiang;
    

    void Start()
    {
        cc = GetComponent<CharacterController>();
        Target = GameObject.Find("Sun");
    }


    void Update()
    {
        cameraView();                                     //这个函数实现了摄像机视角的360度变化
       CameraZoom();                                      //这个函数实现了摄像机视角的缩放
        cameraMove();                                     //这个函数实现了摄像机在空间中的自由移动
    }


    void LateUpdate()
    {
        MouseClick();

        if (Input.GetKey(KeyCode.Escape))

        {

            Application.Quit();

        }
    }


    private void cameraView()
    {
       if (UnityEngine.Input.GetMouseButton(0))                                        //在按住鼠标左键的情况下
        { 
        
        vertiAngle += UnityEngine.Input.GetAxis("Mouse Y");
        horiAngle -= UnityEngine.Input.GetAxis("Mouse X");
           
        
        Quaternion quaternion = Quaternion.Euler(vertiAngle, horiAngle, 0);
        transform.rotation = quaternion;

        distance = (Target.transform.position - transform.position).magnitude;


        tem = new Vector3(0, 0, -distance);

        Vector3 position = quaternion * tem + Target.transform.position;

        transform.position = position;
        }
    }
    
    public void CameraZoom()
    {
        if (UnityEngine.Input.GetMouseButton(0))                                       //在按住鼠标左键的情况下
        {
            Vector3 dir = Camera.main.ScreenPointToRay(UnityEngine.Input.mousePosition).direction;
            transform.position += dir * UnityEngine.Input.GetAxis("Mouse ScrollWheel") * mScrollValue;
        }
    }
    
    private void cameraMove()
    {
        if (UnityEngine.Input.GetKey(KeyCode.Tab))                                     //在按住tab的情况下
        {
            horizontalMove = UnityEngine.Input.GetAxis("Horizontal") * moveSpeed;      //用w和s控制前后
            verticalMove = UnityEngine.Input.GetAxis("Vertical") * moveSpeed;          //用a和d控制左右
            shangxiaMove = UnityEngine.Input.GetAxis("Jump") * moveSpeed;              //用q和e控制上下

            fangxiang = transform.forward * verticalMove + transform.right * horizontalMove + transform.up * shangxiaMove;
            cc.Move(fangxiang * Time.deltaTime);
        }


    }


    void MouseClick()
    {
        //获得摄像头与鼠标位置方向射线的向量
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        //new Vector3(Screen.width/2,Screen.height/2,0);获取屏幕中心点，可应用到上方射线中
        //判断是否点击左键&&射线是否碰到有碰撞器的东西
        if (Input.GetMouseButtonDown(1) && Physics.Raycast(ray, out RaycastHit hitinfo))
        {

            Target2 = Target;

            Target = null;
                      
            Target = GameObject.Find(hitinfo.transform.name);                                                   //实现了鼠标点谁把谁作为中心目标

            Vector3 juli =Target.transform.position - Target2.transform.position;

            transform.LookAt(Target.transform.position);

           // transform.position = Vector3.MoveTowards(transform.position, transform.position + juli,0.0002f * Time.deltaTime);
        }
    }


}
