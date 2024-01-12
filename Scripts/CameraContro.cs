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
        cameraView();                                     //�������ʵ����������ӽǵ�360�ȱ仯
       CameraZoom();                                      //�������ʵ����������ӽǵ�����
        cameraMove();                                     //�������ʵ����������ڿռ��е������ƶ�
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
       if (UnityEngine.Input.GetMouseButton(0))                                        //�ڰ�ס�������������
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
        if (UnityEngine.Input.GetMouseButton(0))                                       //�ڰ�ס�������������
        {
            Vector3 dir = Camera.main.ScreenPointToRay(UnityEngine.Input.mousePosition).direction;
            transform.position += dir * UnityEngine.Input.GetAxis("Mouse ScrollWheel") * mScrollValue;
        }
    }
    
    private void cameraMove()
    {
        if (UnityEngine.Input.GetKey(KeyCode.Tab))                                     //�ڰ�סtab�������
        {
            horizontalMove = UnityEngine.Input.GetAxis("Horizontal") * moveSpeed;      //��w��s����ǰ��
            verticalMove = UnityEngine.Input.GetAxis("Vertical") * moveSpeed;          //��a��d��������
            shangxiaMove = UnityEngine.Input.GetAxis("Jump") * moveSpeed;              //��q��e��������

            fangxiang = transform.forward * verticalMove + transform.right * horizontalMove + transform.up * shangxiaMove;
            cc.Move(fangxiang * Time.deltaTime);
        }


    }


    void MouseClick()
    {
        //�������ͷ�����λ�÷������ߵ�����
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        //new Vector3(Screen.width/2,Screen.height/2,0);��ȡ��Ļ���ĵ㣬��Ӧ�õ��Ϸ�������
        //�ж��Ƿ������&&�����Ƿ���������ײ���Ķ���
        if (Input.GetMouseButtonDown(1) && Physics.Raycast(ray, out RaycastHit hitinfo))
        {

            Target2 = Target;

            Target = null;
                      
            Target = GameObject.Find(hitinfo.transform.name);                                                   //ʵ��������˭��˭��Ϊ����Ŀ��

            Vector3 juli =Target.transform.position - Target2.transform.position;

            transform.LookAt(Target.transform.position);

           // transform.position = Vector3.MoveTowards(transform.position, transform.position + juli,0.0002f * Time.deltaTime);
        }
    }


}
